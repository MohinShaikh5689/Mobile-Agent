/**
 * Models that run on the phone. The user picks one chat model and one embedding model; GGUF files
 * are downloaded once into the app's document directory. Users can also add any GGUF by link.
 *
 * Memory: weights + KV cache + runtime overhead must fit in what the OS lets one app use. On iOS
 * the app carries the increased-memory-limit entitlement; even so, an 8B model on an 8 GB phone
 * is tight (Constraints: "treat it as fragile"). The fit labels below are estimates.
 */

export type ModelRole = 'chat' | 'embed';
export type CacheType = 'f16' | 'q8_0';

export interface ModelSpec {
  id: string;
  role: ModelRole;
  name: string;
  /** Parameter count label, for grouping ("1.7B", "4B", "8B"). */
  params: string;
  url: string;
  fileName: string;
  /** Approximate download size; 0 when unknown (custom models before download). */
  sizeBytes: number;
  /**
   * Context length used when the phone's RAM is unknown. Otherwise the app picks the largest
   * option that fits (autoContext), and the user can change it per model.
   */
  nCtx: number;
  /** The longest context the model was trained for. */
  maxCtx: number;
  /**
   * KV-cache bytes per token at f16: 2 (K and V) × layers × KV heads × head dim × 2 bytes.
   * Used for the memory estimate.
   */
  kvBytesPerToken: number;
  /** KV-cache type. q8_0 halves the cache; the runtime falls back to f16 if the device rejects it. */
  cacheType: CacheType;
  notes: string;
  queryPrefix?: string;
  documentPrefix?: string;
  /** Added by the user from a link. */
  custom?: boolean;
}

const GB = 1024 ** 3;
const MB = 1024 ** 2;

/** 2 × layers × kvHeads × headDim × 2 bytes (f16). */
const kv = (layers: number, kvHeads: number, headDim: number) => 2 * layers * kvHeads * headDim * 2;

export const MODELS: ModelSpec[] = [
  {
    id: 'qwen3-1.7b-q4km',
    role: 'chat',
    name: 'Qwen3 1.7B (Q4_K_M)',
    params: '1.7B',
    url: 'https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf',
    fileName: 'Qwen3-1.7B-Q4_K_M.gguf',
    sizeBytes: 1.1 * GB,
    nCtx: 8192,
    maxCtx: 32768,
    kvBytesPerToken: kv(28, 8, 128),
    cacheType: 'q8_0',
    notes: 'For phones with 6 GB of RAM or less. Fastest, but weaker at multi-step tool use.',
  },
  {
    id: 'qwen3-4b-instruct-2507-q4km',
    role: 'chat',
    name: 'Qwen3 4B Instruct 2507 (Q4_K_M)',
    params: '4B',
    url: 'https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
    fileName: 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
    sizeBytes: 2.5 * GB,
    nCtx: 8192,
    maxCtx: 262144,
    kvBytesPerToken: kv(36, 8, 128),
    cacheType: 'q8_0',
    notes: 'Default. Strongest sub-7B tool caller on the shortlist.',
  },
  {
    id: 'phi-4-mini-instruct-q4km',
    role: 'chat',
    name: 'Phi-4-mini Instruct 3.8B (Q4_K_M)',
    params: '4B',
    url: 'https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf',
    fileName: 'Phi-4-mini-instruct-Q4_K_M.gguf',
    sizeBytes: 2.5 * GB,
    nCtx: 8192,
    maxCtx: 131072,
    kvBytesPerToken: kv(32, 8, 128),
    cacheType: 'q8_0',
    notes: 'Faster, better at planning between calls; somewhat weaker raw tool calls.',
  },
  {
    id: 'qwen3-8b-q4km',
    role: 'chat',
    name: 'Qwen3 8B (Q4_K_M)',
    params: '8B',
    url: 'https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf',
    fileName: 'Qwen3-8B-Q4_K_M.gguf',
    sizeBytes: 5.0 * GB,
    nCtx: 4096,
    maxCtx: 32768,
    kvBytesPerToken: kv(36, 8, 128),
    cacheType: 'q8_0',
    notes: 'Best quality. Needs about 6 GB free: comfortable on 12 GB phones, tight on 8 GB.',
  },
  {
    id: 'nomic-embed-text-v1.5-q8',
    role: 'embed',
    name: 'nomic-embed-text v1.5 (Q8_0)',
    params: '137M',
    url: 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf',
    fileName: 'nomic-embed-text-v1.5.Q8_0.gguf',
    sizeBytes: 146 * MB,
    nCtx: 2048,
    maxCtx: 2048,
    kvBytesPerToken: 0,
    cacheType: 'f16',
    notes: 'Embeddings for the local index. Needed with any chat model.',
    queryPrefix: 'search_query: ',
    documentPrefix: 'search_document: ',
  },
];

export const DEFAULT_CHAT_MODEL = 'qwen3-4b-instruct-2507-q4km';
export const DEFAULT_EMBED_MODEL = 'nomic-embed-text-v1.5-q8';

/** Built-in models plus the user's custom ones. */
export function allModels(custom: ModelSpec[] = []): ModelSpec[] {
  return [...MODELS, ...custom];
}

export function findModel(id: string, custom: ModelSpec[] = []): ModelSpec | undefined {
  return allModels(custom).find((m) => m.id === id);
}

/**
 * A chat model from any direct .gguf link. Assumes a 7–8B-class architecture for the KV estimate,
 * which overestimates smaller models (safe side).
 */
export function customChatModel(name: string, url: string): ModelSpec {
  const clean = url.trim();
  const fileName = decodeURIComponent(clean.split('?')[0]?.split('/').pop() ?? '');
  if (!/^https:\/\//.test(clean)) throw new Error('The link must start with https://');
  if (!fileName.toLowerCase().endsWith('.gguf')) throw new Error('The link must point to a .gguf file.');
  return {
    id: `custom-${fileName.toLowerCase().replace(/[^a-z0-9.]+/g, '-')}`,
    role: 'chat',
    name: name.trim() || fileName,
    params: 'Custom',
    url: clean,
    fileName,
    sizeBytes: 0,
    nCtx: 8192,
    maxCtx: 32768,
    kvBytesPerToken: kv(36, 8, 128),
    cacheType: 'q8_0',
    notes: 'Added from a link. Tool calling depends on the model; context up to 32K is offered, check what it supports.',
    custom: true,
  };
}

/** Context lengths offered in Settings, capped by each model's maxCtx. */
export const CONTEXT_OPTIONS = [4096, 8192, 16384, 32768];

export function contextOptions(spec: ModelSpec): number[] {
  return CONTEXT_OPTIONS.filter((n) => n <= spec.maxCtx);
}

export function withContext(spec: ModelSpec, nCtx: number): ModelSpec {
  return { ...spec, nCtx };
}

/** The largest context that fits this phone comfortably; the smallest option if none does. */
export function autoContext(spec: ModelSpec, totalRam: number | null, sizeBytes = spec.sizeBytes): number {
  const options = contextOptions(spec);
  if (!totalRam || sizeBytes <= 0) return Math.min(spec.nCtx, spec.maxCtx);
  const fitting = options.filter((n) => modelFit(withContext(spec, n), totalRam, sizeBytes) === 'fits');
  return fitting.length ? (fitting[fitting.length - 1] as number) : (options[0] as number);
}

/** The context to run with: the user's choice when it is one of the options, else autoContext. */
export function effectiveContext(spec: ModelSpec, totalRam: number | null, chosen?: number, sizeBytes = spec.sizeBytes): number {
  return chosen && contextOptions(spec).includes(chosen) ? chosen : autoContext(spec, totalRam, sizeBytes);
}

export function formatContext(n: number): string {
  return `${Math.round(n / 1024)}K`;
}

/** Runtime overhead beyond weights and KV cache: compute buffers, Metal/CPU scratch, the app itself. */
const OVERHEAD = 450 * MB;

/** Estimated memory to run a model, in bytes. `sizeBytes` overrides the catalog size (e.g. the real file size). */
export function estimateMemory(spec: ModelSpec, sizeBytes = spec.sizeBytes): number {
  const kvFactor = spec.cacheType === 'q8_0' ? 0.53 : 1; // q8_0 ≈ 8.5 bits per value
  return sizeBytes + spec.kvBytesPerToken * spec.nCtx * kvFactor + OVERHEAD;
}

export type Fit = 'fits' | 'tight' | 'too_big' | 'unknown';

/**
 * How a model fits a phone with `totalRam` bytes. The OS keeps a large share for itself; iOS with
 * the increased-memory-limit entitlement allows roughly 70–80% of RAM to one app.
 */
export function modelFit(spec: ModelSpec, totalRam: number | null, sizeBytes = spec.sizeBytes): Fit {
  if (!totalRam || sizeBytes <= 0) return 'unknown';
  const share = estimateMemory(spec, sizeBytes) / totalRam;
  if (share <= 0.6) return 'fits';
  if (share <= 0.8) return 'tight';
  return 'too_big';
}

/** The largest built-in chat model that fits comfortably; the smallest one if none does. */
export function recommendedChatModel(totalRam: number | null): ModelSpec {
  const chat = MODELS.filter((m) => m.role === 'chat');
  if (!totalRam) return findModel(DEFAULT_CHAT_MODEL) as ModelSpec;
  // A model counts as fitting when it fits with at least an 8K context.
  const fitting = chat.filter((m) => modelFit(withContext(m, 8192), totalRam) === 'fits');
  if (fitting.length === 0) return chat.reduce((a, b) => (a.sizeBytes <= b.sizeBytes ? a : b));
  // Prefer the default among equal-size options (Qwen 4B over Phi-4-mini).
  return fitting.reduce((a, b) =>
    b.sizeBytes > a.sizeBytes || (b.sizeBytes === a.sizeBytes && b.id === DEFAULT_CHAT_MODEL) ? b : a,
  );
}

export function formatBytes(n: number): string {
  if (n >= GB) return `${(n / GB).toFixed(1)} GB`;
  return `${Math.round(n / MB)} MB`;
}
