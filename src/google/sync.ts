import { chunkItem } from '../core/chunker';
import type { Embedder } from '../core/llm/types';
import type { KeyValueStore } from '../core/services';
import type { Chunk, Item, Source } from '../core/types';
import type { VectorIndex } from '../core/vectorIndex';
import type { Coverage } from '../core/watermark';
import type { DriveClient } from './drive';
import { isReadable } from './drive';
import type { GmailClient } from './gmail';
import { GoogleApiError, mapLimit } from './http';

/**
 * Incremental sync (Retrieval design, Sync):
 * - Only new or changed items are embedded. Gmail history ids and the Drive changes feed supply
 *   "what changed since X".
 * - The same feeds remove stale entries (edited or deleted items).
 * - The watermark is set to the time the sync STARTED: anything that arrives during the sync is
 *   fetched live (and de-duplicated if it also got indexed).
 */

export interface SyncOptions {
  /** How far back the first sync reaches. */
  backfillDays: number;
  /** Most items the first sync indexes per source. */
  maxItems: number;
}

export const DEFAULT_SYNC: SyncOptions = { backfillDays: 180, maxItems: 300 };

export interface SyncProgress {
  source: Source;
  phase: 'listing' | 'indexing' | 'done' | 'error';
  done: number;
  total: number;
  message?: string;
}

const DAY = 24 * 60 * 60 * 1000;
const EMBED_BATCH = 16;

const coverageKey = (s: Source) => `coverage.${s}`;
const cursorKey = (s: Source) => `cursor.${s}`;

export async function loadCoverage(kv: KeyValueStore): Promise<Partial<Record<Source, Coverage>>> {
  const out: Partial<Record<Source, Coverage>> = {};
  for (const s of ['mail', 'docs'] as Source[]) {
    const raw = await kv.get(coverageKey(s));
    if (raw) out[s] = JSON.parse(raw) as Coverage;
  }
  return out;
}

export class Indexer {
  constructor(
    private deps: {
      gmail: GmailClient;
      drive: DriveClient;
      index: VectorIndex;
      embedder: Embedder;
      kv: KeyValueStore;
      options?: SyncOptions;
      now?: () => number;
    },
  ) {}

  private get opts(): SyncOptions {
    return this.deps.options ?? DEFAULT_SYNC;
  }

  private now(): number {
    return (this.deps.now ?? Date.now)();
  }

  async syncAll(onProgress?: (p: SyncProgress) => void): Promise<void> {
    for (const s of ['mail', 'docs'] as const) {
      try {
        if (s === 'mail') await this.syncMail(onProgress);
        else await this.syncDocs(onProgress);
      } catch (e) {
        onProgress?.({ source: s, phase: 'error', done: 0, total: 0, message: (e as Error).message });
        throw e;
      }
    }
  }

  /** Forget everything and start over on the next sync. */
  async reset(): Promise<void> {
    await this.deps.index.clear();
    for (const s of ['mail', 'docs'] as Source[]) {
      await this.deps.kv.delete(coverageKey(s));
      await this.deps.kv.delete(cursorKey(s));
    }
  }

  private async indexItems(items: { item: Item; body: string }[]): Promise<void> {
    const pending: Omit<Chunk, 'embedding'>[] = [];
    for (const { item, body } of items) {
      await this.deps.index.removeItem(item.source, item.id);
      pending.push(...chunkItem(item, body));
    }
    for (let i = 0; i < pending.length; i += EMBED_BATCH) {
      const batch = pending.slice(i, i + EMBED_BATCH);
      const vecs = await this.deps.embedder.embed(batch.map((c) => c.text));
      await this.deps.index.upsert(batch.map((c, j) => ({ ...c, embedding: vecs[j] ?? [] })));
    }
  }

  private async setCoverage(source: Source, c: Coverage): Promise<void> {
    await this.deps.kv.set(coverageKey(source), JSON.stringify(c));
  }

  async syncMail(onProgress?: (p: SyncProgress) => void): Promise<void> {
    const { gmail, kv } = this.deps;
    const startedAt = this.now();
    const cursor = await kv.get(cursorKey('mail'));
    const raw = await kv.get(coverageKey('mail'));
    const prev = raw ? (JSON.parse(raw) as Coverage) : null;
    onProgress?.({ source: 'mail', phase: 'listing', done: 0, total: 0 });

    if (cursor && prev) {
      try {
        const { added, deleted, historyId } = await gmail.history(cursor);
        for (const id of deleted) await this.deps.index.removeItem('mail', id);
        await this.indexMessages(added, onProgress);
        await kv.set(cursorKey('mail'), historyId);
        await this.setCoverage('mail', { floor: prev.floor, watermark: startedAt });
        onProgress?.({ source: 'mail', phase: 'done', done: added.length, total: added.length });
        return;
      } catch (e) {
        // A history id that is too old returns 404: fall through to a full resync.
        if (!(e instanceof GoogleApiError && e.status === 404)) throw e;
      }
    }

    const historyId = await gmail.profileHistoryId();
    const floor = startedAt - this.opts.backfillDays * DAY;
    const ids: string[] = [];
    let pageToken: string | undefined;
    do {
      const page = await gmail.listIds(
        `after:${Math.floor(floor / 1000)} -in:spam -in:trash -in:chats`,
        Math.min(100, this.opts.maxItems - ids.length),
        pageToken,
      );
      ids.push(...page.ids);
      pageToken = page.nextPageToken;
    } while (pageToken && ids.length < this.opts.maxItems);

    await this.indexMessages(ids, onProgress);
    // If the cap cut the backfill short, the index is only complete from the oldest message kept.
    const effectiveFloor = ids.length >= this.opts.maxItems ? await this.oldestIndexedMailDate(ids, floor) : floor;
    await kv.set(cursorKey('mail'), historyId);
    await this.setCoverage('mail', { floor: effectiveFloor, watermark: startedAt });
    onProgress?.({ source: 'mail', phase: 'done', done: ids.length, total: ids.length });
  }

  private lastMailDates = new Map<string, number>();

  private async oldestIndexedMailDate(ids: string[], fallback: number): Promise<number> {
    const dates = ids.map((id) => this.lastMailDates.get(id)).filter((d): d is number => d !== undefined);
    return dates.length ? Math.min(...dates) : fallback;
  }

  private async indexMessages(ids: string[], onProgress?: (p: SyncProgress) => void): Promise<void> {
    let done = 0;
    const BATCH = 10;
    for (let i = 0; i < ids.length; i += BATCH) {
      const slice = ids.slice(i, i + BATCH);
      const items = await mapLimit(slice, 4, async (id) => {
        try {
          const item = await this.deps.gmail.read(id);
          this.lastMailDates.set(id, item.date);
          return { item, body: item.text };
        } catch (e) {
          if (e instanceof GoogleApiError && e.status === 404) return null; // deleted meanwhile
          throw e;
        }
      });
      await this.indexItems(items.filter((x): x is { item: Item; body: string } => x !== null));
      done += slice.length;
      onProgress?.({ source: 'mail', phase: 'indexing', done, total: ids.length });
    }
  }

  async syncDocs(onProgress?: (p: SyncProgress) => void): Promise<void> {
    const { drive, kv } = this.deps;
    const startedAt = this.now();
    const cursor = await kv.get(cursorKey('docs'));
    const raw = await kv.get(coverageKey('docs'));
    const prev = raw ? (JSON.parse(raw) as Coverage) : null;
    onProgress?.({ source: 'docs', phase: 'listing', done: 0, total: 0 });

    if (cursor && prev) {
      const { changed, removed, newToken } = await drive.changes(cursor);
      for (const id of removed) await this.deps.index.removeItem('docs', id);
      const readable = changed.filter((f) => isReadable(f.mimeType));
      await this.indexFiles(readable, onProgress);
      await kv.set(cursorKey('docs'), newToken);
      await this.setCoverage('docs', { floor: prev.floor, watermark: startedAt });
      onProgress?.({ source: 'docs', phase: 'done', done: readable.length, total: readable.length });
      return;
    }

    const token = await drive.startPageToken();
    const floor = startedAt - this.opts.backfillDays * DAY;
    const q =
      `trashed = false and modifiedTime >= '${new Date(floor).toISOString()}' and ` +
      "(mimeType = 'application/vnd.google-apps.document' or mimeType contains 'text/')";
    const files: Awaited<ReturnType<DriveClient['listFiles']>>['files'] = [];
    let pageToken: string | undefined;
    do {
      const page = await drive.listFiles(q, 100, pageToken);
      files.push(...page.files);
      pageToken = page.nextPageToken;
    } while (pageToken && files.length < this.opts.maxItems);
    const kept = files.slice(0, this.opts.maxItems);

    await this.indexFiles(kept, onProgress);
    const effectiveFloor =
      files.length > kept.length && kept.length > 0
        ? Math.min(...kept.map((f) => Date.parse(f.modifiedTime)))
        : floor;
    await kv.set(cursorKey('docs'), token);
    await this.setCoverage('docs', { floor: effectiveFloor, watermark: startedAt });
    onProgress?.({ source: 'docs', phase: 'done', done: kept.length, total: kept.length });
  }

  private async indexFiles(
    files: Awaited<ReturnType<DriveClient['listFiles']>>['files'],
    onProgress?: (p: SyncProgress) => void,
  ): Promise<void> {
    let done = 0;
    for (let i = 0; i < files.length; i += 5) {
      const slice = files.slice(i, i + 5);
      const items = await mapLimit(slice, 3, async (f) => {
        const item = await this.deps.drive.readFull(f);
        return { item, body: item.text };
      });
      await this.indexItems(items);
      done += slice.length;
      onProgress?.({ source: 'docs', phase: 'indexing', done, total: files.length });
    }
  }
}
