import * as SQLite from 'expo-sqlite';
import type { KeyValueStore } from '../core/services';
import type { Chunk, Source } from '../core/types';
import type { IndexHit, SearchOptions, VectorIndex } from '../core/vectorIndex';
import { cosine, topK } from '../core/vectorIndex';

/**
 * The local index lives in SQLite on the device. Embeddings are stored as Float32 blobs and
 * scored by brute-force cosine, which is fast enough for a few thousand chunks of personal data.
 * A vector extension (sqlite-vec) can replace the scan later without changing the VectorIndex API.
 */

let dbPromise: Promise<SQLite.SQLiteDatabase> | null = null;

export function openDb(): Promise<SQLite.SQLiteDatabase> {
  dbPromise ??= (async () => {
    const db = await SQLite.openDatabaseAsync('agent.db');
    await db.execAsync(`
      PRAGMA journal_mode = WAL;
      CREATE TABLE IF NOT EXISTS chunks (
        source TEXT NOT NULL,
        id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        title TEXT NOT NULL,
        sender TEXT,
        date INTEGER NOT NULL,
        text TEXT NOT NULL,
        embedding BLOB NOT NULL,
        PRIMARY KEY (source, id)
      );
      CREATE INDEX IF NOT EXISTS chunks_item ON chunks (source, item_id);
      CREATE INDEX IF NOT EXISTS chunks_date ON chunks (source, date);
      CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    `);
    return db;
  })();
  return dbPromise;
}

function toBlob(v: ArrayLike<number>): Uint8Array {
  return new Uint8Array(Float32Array.from(v).buffer);
}

function fromBlob(b: Uint8Array): Float32Array {
  // Copy so the Float32Array starts on a 4-byte boundary.
  return new Float32Array(Uint8Array.from(b).buffer);
}

interface ChunkRow {
  source: Source;
  id: string;
  item_id: string;
  title: string;
  sender: string | null;
  date: number;
  text: string;
  embedding: Uint8Array;
}

export class SqliteVectorIndex implements VectorIndex {
  async upsert(chunks: Chunk[]): Promise<void> {
    const db = await openDb();
    await db.withTransactionAsync(async () => {
      for (const c of chunks) {
        await db.runAsync(
          `INSERT OR REPLACE INTO chunks (source, id, item_id, title, sender, date, text, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
          [c.source, c.id, c.itemId, c.title, c.from ?? null, c.date, c.text, toBlob(c.embedding)],
        );
      }
    });
  }

  async removeItem(source: Source, itemId: string): Promise<void> {
    const db = await openDb();
    await db.runAsync('DELETE FROM chunks WHERE source = ? AND item_id = ?', [source, itemId]);
  }

  async search(query: number[], opts: SearchOptions): Promise<IndexHit[]> {
    if (opts.sources.length === 0) return [];
    const db = await openDb();
    const where = [`source IN (${opts.sources.map(() => '?').join(',')})`];
    const params: (string | number)[] = [...opts.sources];
    if (opts.window?.start !== undefined) {
      where.push('date >= ?');
      params.push(opts.window.start);
    }
    if (opts.window?.end !== undefined) {
      where.push('date < ?');
      params.push(opts.window.end);
    }
    const hits: IndexHit[] = [];
    // Stream rows so the whole index never sits in memory at once.
    for await (const row of db.getEachAsync<ChunkRow>(`SELECT * FROM chunks WHERE ${where.join(' AND ')}`, params)) {
      hits.push({
        score: cosine(query, fromBlob(row.embedding)),
        chunk: {
          id: row.id,
          itemId: row.item_id,
          source: row.source,
          title: row.title,
          from: row.sender ?? undefined,
          date: row.date,
          text: row.text,
          embedding: [],
        },
      });
    }
    return topK(hits, opts.k);
  }

  async count(): Promise<number> {
    const db = await openDb();
    const row = await db.getFirstAsync<{ n: number }>('SELECT COUNT(*) AS n FROM chunks');
    return row?.n ?? 0;
  }

  async clear(): Promise<void> {
    const db = await openDb();
    await db.runAsync('DELETE FROM chunks');
  }
}

export class SqliteKeyValueStore implements KeyValueStore {
  async get(key: string): Promise<string | null> {
    const db = await openDb();
    const row = await db.getFirstAsync<{ value: string }>('SELECT value FROM kv WHERE key = ?', [key]);
    return row?.value ?? null;
  }

  async set(key: string, value: string): Promise<void> {
    const db = await openDb();
    await db.runAsync('INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)', [key, value]);
  }

  async delete(key: string): Promise<void> {
    const db = await openDb();
    await db.runAsync('DELETE FROM kv WHERE key = ?', [key]);
  }
}
