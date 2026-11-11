import { DurableCheckpointSaver } from './core/graph/checkpointer';
import type { OnDeviceModel } from './core/llm/onDevice';
import { Agent } from './core/orchestrator';
import { HybridRetriever } from './core/retriever';
import type { Services } from './core/services';
import { getAccessToken } from './google/auth';
import { CalendarClient } from './google/calendar';
import { DriveClient } from './google/drive';
import { GmailClient } from './google/gmail';
import { Indexer, loadCoverage } from './google/sync';
import { getRuntime } from './llm/runtime';
import type { Settings } from './storage/settings';
import { SqliteKeyValueStore, SqliteVectorIndex } from './storage/sqlite';

export const kv = new SqliteKeyValueStore();
export const vectorIndex = new SqliteVectorIndex();
/** Graph checkpoints live in the same on-device SQLite database. */
export const checkpointer = new DurableCheckpointSaver(kv);

export interface AppContainer {
  agent: Agent;
  indexer: Indexer;
  model: OnDeviceModel;
}

export type ContainerResult = { ok: true; container: AppContainer } | { ok: false; reason: string };

/** Builds the agent graph around the on-device model. Rebuilt whenever settings change. */
export async function buildContainer(settings: Settings): Promise<ContainerResult> {
  const runtime = await getRuntime(settings.chatModel, settings.embedModel);
  if (!runtime.ok) return { ok: false, reason: runtime.reason };
  const model = runtime.model;

  const gmail = new GmailClient(getAccessToken);
  const drive = new DriveClient(getAccessToken);
  const calendar = new CalendarClient(getAccessToken);
  const services: Services = { mail: gmail, docs: drive, calendar };

  const retriever = new HybridRetriever({
    index: vectorIndex,
    embedder: model,
    coverage: () => loadCoverage(kv),
    live: {
      mail: ({ keywords, window, max }) => gmail.search({ keywords, window, max }),
      docs: ({ keywords, window, max }) => drive.search({ keywords, window, max }),
    },
  });

  return {
    ok: true,
    container: {
      agent: new Agent({
        model,
        retriever,
        services,
        loadCoverage: () => loadCoverage(kv),
        checkpointer,
        kv,
        maxSteps: settings.maxSteps,
      }),
      indexer: new Indexer({ gmail, drive, index: vectorIndex, embedder: model, kv }),
      model,
    },
  };
}
