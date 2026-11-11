import type { ChatMessage, ChatModel, ChatResult, EmbedKind, Embedder, ToolCall, ToolSpec } from './types';
import { parseJsonObject, stripThinking } from './types';

/**
 * The model runs on the phone through llama.rn (llama.cpp). This file holds the mapping between
 * the agent's ChatModel/Embedder interfaces and llama.rn, with the native module injected so it
 * can be unit-tested. src/llm/runtime.ts passes the real `initLlama`.
 */

/** The slice of llama.rn's LlamaContext this backend uses. */
export interface LlamaLike {
  completion(params: Record<string, unknown>): Promise<{
    text: string;
    content?: string;
    tool_calls?: { type?: string; id?: string; function: { name: string; arguments: string } }[];
  }>;
  embedding(text: string): Promise<{ embedding: number[] }>;
  release(): Promise<void>;
}

export type InitFn = (params: Record<string, unknown>, onProgress?: (p: number) => void) => Promise<LlamaLike>;

export interface OnDeviceConfig {
  chatModelPath: string;
  embedModelPath: string;
  chatCtx: number;
  embedCtx: number;
  queryPrefix?: string;
  documentPrefix?: string;
  /** Upper bound on generated tokens per call. */
  maxTokens?: number;
}

/** One completion at a time per context: llama.cpp contexts are not re-entrant. */
class SerialQueue {
  private tail: Promise<unknown> = Promise.resolve();
  run<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.tail.then(fn, fn);
    this.tail = next.catch(() => undefined);
    return next;
  }
}

/** OpenAI-style messages, which llama.rn renders with the model's own Jinja chat template. */
export function toOaiMessages(messages: ChatMessage[]): Record<string, unknown>[] {
  let callCounter = 0;
  const pendingIds: string[] = [];
  return messages.map((m) => {
    if (m.role === 'assistant' && m.toolCalls?.length) {
      const tool_calls = m.toolCalls.map((c) => {
        const id = `call_${callCounter++}`;
        pendingIds.push(id);
        return { type: 'function', id, function: { name: c.name, arguments: JSON.stringify(c.arguments) } };
      });
      return { role: 'assistant', content: m.content ?? '', tool_calls };
    }
    if (m.role === 'tool') {
      return { role: 'tool', content: m.content, name: m.toolName, tool_call_id: pendingIds.shift() ?? `call_${callCounter++}` };
    }
    return { role: m.role, content: m.content };
  });
}

function parseArgs(raw: string): Record<string, unknown> {
  try {
    const v = JSON.parse(raw);
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

export class OnDeviceModel implements ChatModel, Embedder {
  private chatCtx: Promise<LlamaLike> | null = null;
  private embedCtx: Promise<LlamaLike> | null = null;
  private chatQueue = new SerialQueue();
  private embedQueue = new SerialQueue();

  constructor(
    private cfg: OnDeviceConfig,
    private init: InitFn,
    private onLoadProgress?: (which: 'chat' | 'embed', progress: number) => void,
  ) {}

  /** Loaded on first use, so the app starts fast and the model is mapped only when needed. */
  private chatContext(): Promise<LlamaLike> {
    this.chatCtx ??= this.init(
      {
        model: this.cfg.chatModelPath,
        n_ctx: this.cfg.chatCtx,
        n_batch: 512,
        // Offload every layer to Metal on iOS; ignored where no GPU backend is available.
        n_gpu_layers: 99,
        use_mmap: true,
        use_mlock: false,
        flash_attn_type: 'auto',
        ctx_shift: false,
      },
      (p) => this.onLoadProgress?.('chat', p),
    ).catch((e) => {
      this.chatCtx = null;
      throw e;
    });
    return this.chatCtx;
  }

  private embedContext(): Promise<LlamaLike> {
    this.embedCtx ??= this.init(
      {
        model: this.cfg.embedModelPath,
        embedding: true,
        pooling_type: 'mean',
        n_ctx: this.cfg.embedCtx,
        n_batch: this.cfg.embedCtx,
        n_ubatch: this.cfg.embedCtx,
        n_gpu_layers: 99,
        use_mmap: true,
      },
      (p) => this.onLoadProgress?.('embed', p),
    ).catch((e) => {
      this.embedCtx = null;
      throw e;
    });
    return this.embedCtx;
  }

  async chat(messages: ChatMessage[], tools?: ToolSpec[]): Promise<ChatResult> {
    const ctx = await this.chatContext();
    const res = await this.chatQueue.run(() =>
      ctx.completion({
        messages: toOaiMessages(messages),
        jinja: true,
        ...(tools?.length
          ? { tools: tools.map((t) => ({ type: 'function', function: t })), tool_choice: 'auto', parallel_tool_calls: false }
          : {}),
        // Deterministic output makes eval runs comparable.
        temperature: 0,
        n_predict: this.cfg.maxTokens ?? 1024,
        // Hybrid-thinking models (Qwen3) answer directly; the graph does the planning.
        chat_template_kwargs: { enable_thinking: false },
      }),
    );
    const toolCalls: ToolCall[] = (res.tool_calls ?? []).map((c) => ({
      name: c.function.name,
      arguments: parseArgs(c.function.arguments),
    }));
    return { content: stripThinking(res.content ?? res.text ?? ''), toolCalls };
  }

  /** Grammar-constrained JSON: llama.cpp turns the schema into a grammar, so output always parses. */
  async chatJson(messages: ChatMessage[], schema: Record<string, unknown>): Promise<unknown> {
    const ctx = await this.chatContext();
    const res = await this.chatQueue.run(() =>
      ctx.completion({
        messages: toOaiMessages(messages),
        jinja: true,
        response_format: { type: 'json_schema', json_schema: { strict: true, schema } },
        temperature: 0,
        n_predict: 512,
        chat_template_kwargs: { enable_thinking: false },
      }),
    );
    return parseJsonObject(res.content || res.text);
  }

  async embed(texts: string[], kind: EmbedKind = 'document'): Promise<number[][]> {
    if (texts.length === 0) return [];
    const ctx = await this.embedContext();
    const prefix = (kind === 'query' ? this.cfg.queryPrefix : this.cfg.documentPrefix) ?? '';
    // Roughly 3 characters per token keeps a chunk inside the embedding context.
    const maxChars = Math.floor(this.cfg.embedCtx * 3) - prefix.length;
    return this.embedQueue.run(async () => {
      const out: number[][] = [];
      for (const t of texts) out.push((await ctx.embedding(prefix + t.slice(0, maxChars))).embedding);
      return out;
    });
  }

  /** Frees both models' memory. */
  async release(): Promise<void> {
    const ctxs = [this.chatCtx, this.embedCtx];
    this.chatCtx = null;
    this.embedCtx = null;
    for (const c of ctxs) {
      if (!c) continue;
      try {
        await (await c).release();
      } catch {
        // already gone
      }
    }
  }
}
