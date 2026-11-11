import type { RunnableConfig } from '@langchain/core/runnables';
import type { Checkpoint, CheckpointMetadata, CheckpointTuple } from '@langchain/langgraph/web';
import { MemorySaver } from '@langchain/langgraph/web';
import { base64ToBytes, bytesToBase64 } from '../encoding';
import type { KeyValueStore } from '../services';

/**
 * A checkpointer that survives app restarts (Agents and orchestration: "a checkpointer saves state
 * at every step, so a run can pause for approval and resume, and a crash loses nothing").
 *
 * It reuses LangGraph's own MemorySaver for all checkpoint logic and writes each thread through to
 * a key-value store (SQLite on the device) after every change. A thread is loaded back lazily the
 * first time it is touched after a restart.
 */

type Blob = Uint8Array;
type CheckpointListOptions = Parameters<MemorySaver['list']>[1];
type StoredCheckpoint = [Blob, Blob, string | undefined];
type StoredWrite = [string, string, Blob, string?];

interface ThreadRecord {
  storage: Record<string, Record<string, [string, string, string | undefined]>>;
  writes: Record<string, Record<string, [string, string, string, string?]>>;
}

const PREFIX = 'checkpoint:';

function threadOf(config: RunnableConfig): string | undefined {
  const id = config.configurable?.thread_id;
  return typeof id === 'string' ? id : undefined;
}

/** MemorySaver keys `writes` by JSON.stringify([threadId, ns, checkpointId]). */
function writesKeyThread(key: string): string | undefined {
  try {
    return (JSON.parse(key) as unknown[])[0] as string;
  } catch {
    return undefined;
  }
}

export class DurableCheckpointSaver extends MemorySaver {
  private loaded = new Set<string>();
  private saving = new Map<string, Promise<void>>();

  constructor(private kv: KeyValueStore) {
    super();
  }

  private async ensureLoaded(threadId: string | undefined): Promise<void> {
    if (!threadId || this.loaded.has(threadId)) return;
    this.loaded.add(threadId);
    const raw = await this.kv.get(PREFIX + threadId);
    if (!raw) return;
    const rec = JSON.parse(raw) as ThreadRecord;
    const storage: Record<string, Record<string, StoredCheckpoint>> = {};
    for (const [ns, byId] of Object.entries(rec.storage)) {
      storage[ns] = {};
      for (const [id, [cp, meta, parent]] of Object.entries(byId)) {
        storage[ns][id] = [base64ToBytes(cp), base64ToBytes(meta), parent];
      }
    }
    this.storage[threadId] = storage;
    for (const [key, inner] of Object.entries(rec.writes)) {
      const out: Record<string, StoredWrite> = {};
      for (const [k, [task, channel, value, path]] of Object.entries(inner)) {
        out[k] = [task, channel, base64ToBytes(value), path];
      }
      this.writes[key] = out;
    }
  }

  private async persist(threadId: string): Promise<void> {
    // Serialize saves per thread so an older snapshot never overwrites a newer one.
    const prev = this.saving.get(threadId) ?? Promise.resolve();
    const next = prev.then(async () => {
      const rec: ThreadRecord = { storage: {}, writes: {} };
      for (const [ns, byId] of Object.entries(this.storage[threadId] ?? {})) {
        rec.storage[ns] = {};
        for (const [id, [cp, meta, parent]] of Object.entries(byId)) {
          rec.storage[ns][id] = [bytesToBase64(cp), bytesToBase64(meta), parent];
        }
      }
      for (const [key, inner] of Object.entries(this.writes)) {
        if (writesKeyThread(key) !== threadId) continue;
        rec.writes[key] = {};
        for (const [k, [task, channel, value, path]] of Object.entries(inner)) {
          rec.writes[key][k] = [task, channel, bytesToBase64(value), path];
        }
      }
      await this.kv.set(PREFIX + threadId, JSON.stringify(rec));
    });
    this.saving.set(threadId, next.catch(() => undefined));
    return next;
  }

  override async getTuple(config: RunnableConfig): Promise<CheckpointTuple | undefined> {
    await this.ensureLoaded(threadOf(config));
    return super.getTuple(config);
  }

  override async *list(config: RunnableConfig, options?: CheckpointListOptions): AsyncGenerator<CheckpointTuple> {
    await this.ensureLoaded(threadOf(config));
    yield* super.list(config, options);
  }

  override async put(config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata): Promise<RunnableConfig> {
    const threadId = threadOf(config);
    await this.ensureLoaded(threadId);
    const out = await super.put(config, checkpoint, metadata);
    if (threadId) await this.persist(threadId);
    return out;
  }

  override async putWrites(
    config: RunnableConfig,
    writes: Parameters<MemorySaver['putWrites']>[1],
    taskId: string,
    taskPath?: string,
  ): Promise<void> {
    const threadId = threadOf(config);
    await this.ensureLoaded(threadId);
    await super.putWrites(config, writes, taskId, taskPath);
    if (threadId) await this.persist(threadId);
  }

  override async deleteThread(threadId: string): Promise<void> {
    await super.deleteThread(threadId);
    this.loaded.delete(threadId);
    await this.kv.delete(PREFIX + threadId);
  }
}
