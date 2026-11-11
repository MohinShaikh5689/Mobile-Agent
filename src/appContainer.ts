import { OllamaChatModel } from './core/llm/ollama';
import { Agent } from './core/orchestrator';
import { HybridRetriever } from './core/retriever';
import type { Services } from './core/services';
import { getAccessToken } from './google/auth';
import { CalendarClient } from './google/calendar';
import { DriveClient } from './google/drive';
import { GmailClient } from './google/gmail';
import { Indexer, loadCoverage } from './google/sync';
import type { Settings } from './storage/settings';
import { SqliteKeyValueStore, SqliteVectorIndex } from './storage/sqlite';

export const kv = new SqliteKeyValueStore();
export const vectorIndex = new SqliteVectorIndex();

export interface AppContainer {
  agent: Agent;
  indexer: Indexer;
  model: OllamaChatModel;
}

/** Builds the agent graph from the current settings. Rebuilt whenever settings change. */
export function buildContainer(settings: Settings): AppContainer {
  const model = new OllamaChatModel({
    baseUrl: settings.ollamaUrl,
    chatModel: settings.chatModel,
    embedModel: settings.embedModel,
  });
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
    agent: new Agent({ model, retriever, services, maxSteps: settings.maxSteps }),
    indexer: new Indexer({ gmail, drive, index: vectorIndex, embedder: model, kv }),
    model,
  };
}
