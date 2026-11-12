import type { ChatMessage, ChatModel, ChatOptions, ChatResult, EmbedKind, Embedder, ToolCall, ToolSpec } from './types';
import { parseJsonObject, stripThinking } from './types';

/**
 * The model runs on the phone through llama.rn (llama.cpp). This file holds the mapping between
 * the agent's ChatModel/Embedder interfaces and llama.rn, with the native module injected so it
 * can be unit-tested. src/llm/runtime.ts passes the real `initLlama`.
 */

/** The slice of llama.rn's LlamaContext this backend uses. */
/** What llama.rn reports for each generated token. `content` is the parsed answer so far. */
export interface TokenEvent {
  token?: string;
  content?: string;
  accumulated_text?: string;
  tool_calls?: unknown[];
}

export interface LlamaLike {
  completion(params: Record<string, unknown>, onToken?: (data: TokenEvent) => void): Promise<{
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
  /** KV-cache type for the chat model. q8_0 halves its memory; falls back to f16 if the device rejects it. */
  chatCacheType?: 'f16' | 'q8_0';
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

/**
 * Turns llama.rn token events into "visible answer so far". Prefers the parsed `content` (tool
 * calls and reasoning already removed); falls back to the raw text when a template gives none,
 * holding back anything that looks like a tool call.
 */
export function partialReporter(onPartial: (text: string) => void): (data: TokenEvent) => void {
  let last = '';
  let toolTurn = false;
  return (data) => {
    if (toolTurn) return;
    if (data.tool_calls?.length) {
      toolTurn = true;
      if (last) onPartial('');
      last = '';
      return;
    }
    let text = data.content;
    if (text === undefined) {
      const raw = data.accumulated_text ?? '';
      if (/<tool_call>|^\s*[{[]/.test(raw)) return;
      text = raw;
    }
    text = stripThinking(text);
    if (text !== last) {
      last = text;
      onPartial(text);
    }
  };
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
    this.chatCtx ??= this.loadChat().catch((e) => {
      this.chatCtx = null;
      throw e;
    });
    return this.chatCtx;
  }

  private async loadChat(): Promise<LlamaLike> {
    const base = {
      model: this.cfg.chatModelPath,
      n_ctx: this.cfg.chatCtx,
      n_batch: 512,
      // Offload every layer to Metal on iOS; ignored where no GPU backend is available.
      n_gpu_layers: 99,
      use_mmap: true,
      use_mlock: false,
      ctx_shift: false,
    };
    return this.withLoadProgress('chat', async (progress) => {
      if (this.cfg.chatCacheType === 'q8_0') {
        try {
          // A quantized V cache needs flash attention.
          return await this.init({ ...base, flash_attn_type: 'on', cache_type_k: 'q8_0', cache_type_v: 'q8_0' }, progress);
        } catch {
          // Some devices or backends reject it: fall through to the plain f16 cache.
        }
      }
      return this.init({ ...base, flash_attn_type: 'auto' }, progress);
    });
  }

  /**
   * Reports load progress only while a load is running, and always ends with 100 (also on
   * failure). Native progress events arrive asynchronously and can land after the load finished;
   * those are dropped, so a stale "Loading…" never sticks.
   */
  private async withLoadProgress<T>(which: 'chat' | 'embed', run: (progress: (p: number) => void) => Promise<T>): Promise<T> {
    let finished = false;
    const progress = (p: number) => {
      if (!finished) this.onLoadProgress?.(which, Math.min(p, 99));
    };
    try {
      return await run(progress);
    } finally {
      finished = true;
      this.onLoadProgress?.(which, 100);
    }
  }

  private embedContext(): Promise<LlamaLike> {
    this.embedCtx ??= this.withLoadProgress('embed', (progress) =>
      this.init(
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
        progress,
      ),
    ).catch((e) => {
      this.embedCtx = null;
      throw e;
    });
    return this.embedCtx;
  }

  async chat(messages: ChatMessage[], tools?: ToolSpec[], opts?: ChatOptions): Promise<ChatResult> {
    const ctx = await this.chatContext();
    const onToken = opts?.onPartial ? partialReporter(opts.onPartial) : undefined;
    const res = await this.chatQueue.run(() =>
      ctx.completion({
        messages: toOaiMessages(messages),
        jinja: true,
        ...(tools?.length
          ? {
              tools: tools.map((t) => ({ type: 'function', function: t })),
              tool_choice: opts?.toolChoice ?? 'auto',
              parallel_tool_calls: false,
            }
          : {}),
        // Deterministic output makes eval runs comparable.
        temperature: 0,
        n_predict: this.cfg.maxTokens ?? 1024,
        // Hybrid-thinking models (Qwen3) answer directly; the graph does the planning.
        chat_template_kwargs: { enable_thinking: false },
      }, onToken),
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

  /**
   * Frees both models' memory. Waits for a running completion or embedding to finish first, so a
   * context is never freed while native code is using it.
   */
  async release(): Promise<void> {
    const chat = this.chatCtx;
    const embed = this.embedCtx;
    this.chatCtx = null;
    this.embedCtx = null;
    const free = async (c: Promise<LlamaLike> | null) => {
      if (!c) return;
      try {
        await (await c).release();
      } catch {
        // already gone, or never loaded
      }
    };
    await Promise.all([this.chatQueue.run(() => free(chat)), this.embedQueue.run(() => free(embed))]);
  }

}
