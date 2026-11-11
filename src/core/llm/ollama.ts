import type { ChatMessage, ChatModel, ChatResult, Embedder, ToolCall, ToolSpec } from './types';
import { parseJsonObject, stripThinking } from './types';

export interface OllamaConfig {
  /** e.g. http://192.168.1.20:11434 — the laptop running `ollama serve` with OLLAMA_HOST=0.0.0.0 */
  baseUrl: string;
  chatModel: string;
  embedModel: string;
  /** Context stays at 8–16K tokens (Constraints). */
  numCtx?: number;
  timeoutMs?: number;
}

export class ModelError extends Error {}

type FetchFn = typeof fetch;

async function postJson<T>(fetchFn: FetchFn, url: string, body: unknown, timeoutMs: number): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetchFn(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new ModelError(`Model server returned ${res.status}: ${text.slice(0, 200)}`);
    }
    return (await res.json()) as T;
  } catch (e) {
    if (e instanceof ModelError) throw e;
    const msg = (e as Error).name === 'AbortError' ? 'timed out' : (e as Error).message;
    throw new ModelError(`Could not reach the model server at ${url} (${msg}).`);
  } finally {
    clearTimeout(timer);
  }
}

interface OllamaChatResponse {
  message?: {
    content?: string;
    tool_calls?: { function: { name: string; arguments: Record<string, unknown> | string } }[];
  };
}

function toOllamaMessage(m: ChatMessage): Record<string, unknown> {
  const out: Record<string, unknown> = { role: m.role, content: m.content };
  if (m.toolCalls?.length) {
    out.tool_calls = m.toolCalls.map((c) => ({ function: { name: c.name, arguments: c.arguments } }));
  }
  if (m.role === 'tool' && m.toolName) out.tool_name = m.toolName;
  return out;
}

function parseArgs(a: Record<string, unknown> | string): Record<string, unknown> {
  if (typeof a !== 'string') return a ?? {};
  try {
    const v = JSON.parse(a);
    return v && typeof v === 'object' ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

export class OllamaChatModel implements ChatModel, Embedder {
  constructor(
    private cfg: OllamaConfig,
    private fetchFn: FetchFn = fetch,
  ) {}

  private url(path: string): string {
    return `${this.cfg.baseUrl.replace(/\/+$/, '')}${path}`;
  }

  async chat(messages: ChatMessage[], tools?: ToolSpec[]): Promise<ChatResult> {
    const body: Record<string, unknown> = {
      model: this.cfg.chatModel,
      messages: messages.map(toOllamaMessage),
      stream: false,
      // Deterministic output makes eval runs comparable.
      options: { temperature: 0, num_ctx: this.cfg.numCtx ?? 8192 },
    };
    if (tools?.length) {
      body.tools = tools.map((t) => ({ type: 'function', function: t }));
    }
    const res = await postJson<OllamaChatResponse>(this.fetchFn, this.url('/api/chat'), body, this.cfg.timeoutMs ?? 120_000);
    const toolCalls: ToolCall[] = (res.message?.tool_calls ?? []).map((c) => ({
      name: c.function.name,
      arguments: parseArgs(c.function.arguments),
    }));
    return { content: stripThinking(res.message?.content ?? ''), toolCalls };
  }

  async chatJson(messages: ChatMessage[], schema: Record<string, unknown>): Promise<unknown> {
    // Ollama constrains decoding to the schema via `format`.
    const res = await postJson<OllamaChatResponse>(
      this.fetchFn,
      this.url('/api/chat'),
      {
        model: this.cfg.chatModel,
        messages: messages.map(toOllamaMessage),
        stream: false,
        format: schema,
        options: { temperature: 0, num_ctx: this.cfg.numCtx ?? 8192 },
      },
      this.cfg.timeoutMs ?? 120_000,
    );
    return parseJsonObject(res.message?.content ?? '');
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const res = await postJson<{ embeddings?: number[][] }>(
      this.fetchFn,
      this.url('/api/embed'),
      { model: this.cfg.embedModel, input: texts },
      this.cfg.timeoutMs ?? 120_000,
    );
    if (!res.embeddings || res.embeddings.length !== texts.length) {
      throw new ModelError('Embedding response did not match the input.');
    }
    return res.embeddings;
  }

  /** Quick reachability check for the settings screen. */
  async listModels(): Promise<string[]> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 5000);
    try {
      const res = await this.fetchFn(this.url('/api/tags'), { signal: ctrl.signal });
      const json = (await res.json()) as { models?: { name: string }[] };
      return (json.models ?? []).map((m) => m.name);
    } catch (e) {
      throw new ModelError(`Could not reach ${this.cfg.baseUrl}: ${(e as Error).message}`);
    } finally {
      clearTimeout(timer);
    }
  }
}
