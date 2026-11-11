/**
 * Models that run on the phone. GGUF files are downloaded once into the app's document directory.
 *
 * Sizing (Constraints): a 3–4B model at Q4 fits on both targets. On iOS the app carries the
 * increased-memory-limit entitlement so a ~2.5 GB model plus its KV cache stays under the cap.
 * The download URL can be changed in Settings if a file moves.
 */

export type ModelRole = 'chat' | 'embed';

export interface ModelSpec {
  id: string;
  role: ModelRole;
  name: string;
  url: string;
  fileName: string;
  /** Approximate download size, for the UI and the free-space check. */
  sizeBytes: number;
  /** Context length the app runs it at. */
  nCtx: number;
  notes: string;
  /** Embedding prefixes, for models trained with them. */
  queryPrefix?: string;
  documentPrefix?: string;
}

const GB = 1024 ** 3;
const MB = 1024 ** 2;

export const MODELS: ModelSpec[] = [
  {
    id: 'qwen3-4b-instruct-2507-q4km',
    role: 'chat',
    name: 'Qwen3 4B Instruct 2507 (Q4_K_M)',
    url: 'https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
    fileName: 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
    sizeBytes: 2.5 * GB,
    nCtx: 8192,
    notes: 'Default. Strongest sub-7B tool caller on the shortlist.',
  },
  {
    id: 'phi-4-mini-instruct-q4km',
    role: 'chat',
    name: 'Phi-4-mini Instruct 3.8B (Q4_K_M)',
    url: 'https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf',
    fileName: 'Phi-4-mini-instruct-Q4_K_M.gguf',
    sizeBytes: 2.5 * GB,
    nCtx: 8192,
    notes: 'Faster, better at planning between calls; somewhat weaker raw tool calls.',
  },
  {
    id: 'nomic-embed-text-v1.5-q8',
    role: 'embed',
    name: 'nomic-embed-text v1.5 (Q8_0)',
    url: 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf',
    fileName: 'nomic-embed-text-v1.5.Q8_0.gguf',
    sizeBytes: 146 * MB,
    nCtx: 2048,
    notes: 'Embeddings for the local index.',
    queryPrefix: 'search_query: ',
    documentPrefix: 'search_document: ',
  },
];

export const DEFAULT_CHAT_MODEL = 'qwen3-4b-instruct-2507-q4km';
export const DEFAULT_EMBED_MODEL = 'nomic-embed-text-v1.5-q8';

export function findModel(id: string): ModelSpec | undefined {
  return MODELS.find((m) => m.id === id);
}

export function formatBytes(n: number): string {
  if (n >= GB) return `${(n / GB).toFixed(1)} GB`;
  return `${Math.round(n / MB)} MB`;
}
