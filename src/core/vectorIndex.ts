import type { Chunk, Source, TimeWindow } from './types';
import { inWindow } from './watermark';

export interface IndexHit {
  chunk: Chunk;
  score: number;
}

export interface SearchOptions {
  sources: Source[];
  window: TimeWindow | null;
  k: number;
}

/** The local vector index. The app uses SQLite (src/storage/sqliteIndex.ts); tests use the in-memory one. */
export interface VectorIndex {
  upsert(chunks: Chunk[]): Promise<void>;
  /** Remove every chunk of an item (edited or deleted upstream). */
  removeItem(source: Source, itemId: string): Promise<void>;
  search(query: number[], opts: SearchOptions): Promise<IndexHit[]>;
  count(): Promise<number>;
  clear(): Promise<void>;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** Keeps the k best hits without sorting the whole candidate set. */
export function topK<T extends { score: number }>(items: Iterable<T>, k: number): T[] {
  const out: T[] = [];
  for (const it of items) {
    if (out.length < k) {
      out.push(it);
      out.sort((a, b) => b.score - a.score);
    } else if (it.score > (out[out.length - 1] as T).score) {
      out[out.length - 1] = it;
      out.sort((a, b) => b.score - a.score);
    }
  }
  return out;
}

export class InMemoryVectorIndex implements VectorIndex {
  private chunks = new Map<string, Chunk>();

  async upsert(chunks: Chunk[]): Promise<void> {
    for (const c of chunks) this.chunks.set(`${c.source}:${c.id}`, c);
  }

  async removeItem(source: Source, itemId: string): Promise<void> {
    for (const [key, c] of this.chunks) {
      if (c.source === source && c.itemId === itemId) this.chunks.delete(key);
    }
  }

  async search(query: number[], opts: SearchOptions): Promise<IndexHit[]> {
    const candidates: IndexHit[] = [];
    for (const c of this.chunks.values()) {
      if (!opts.sources.includes(c.source) || !inWindow(c.date, opts.window)) continue;
      candidates.push({ chunk: c, score: cosine(query, c.embedding) });
    }
    return topK(candidates, opts.k);
  }

  async count(): Promise<number> {
    return this.chunks.size;
  }

  async clear(): Promise<void> {
    this.chunks.clear();
  }
}
