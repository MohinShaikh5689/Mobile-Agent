import { DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, findModel } from '../core/llm/models';
import type { KeyValueStore } from '../core/services';

export interface Settings {
  /** Model ids from src/core/llm/models.ts. Both run on the phone. */
  chatModel: string;
  embedModel: string;
  /** Per-model download URL overrides, in case a file moves. */
  modelUrls: Record<string, string>;
  maxSteps: number;
}

export const DEFAULT_SETTINGS: Settings = {
  chatModel: DEFAULT_CHAT_MODEL,
  embedModel: DEFAULT_EMBED_MODEL,
  modelUrls: {},
  maxSteps: 5,
};

const KEY = 'settings';

export async function loadSettings(kv: KeyValueStore): Promise<Settings> {
  const raw = await kv.get(KEY);
  if (!raw) return DEFAULT_SETTINGS;
  try {
    const parsed = JSON.parse(raw) as Partial<Settings>;
    // Ids not in the catalog (such as Ollama names saved by older builds) fall back to the defaults.
    return {
      chatModel: findModel(parsed.chatModel ?? '')?.role === 'chat' ? (parsed.chatModel as string) : DEFAULT_SETTINGS.chatModel,
      embedModel: findModel(parsed.embedModel ?? '')?.role === 'embed' ? (parsed.embedModel as string) : DEFAULT_SETTINGS.embedModel,
      modelUrls: parsed.modelUrls ?? {},
      maxSteps: parsed.maxSteps ?? DEFAULT_SETTINGS.maxSteps,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export async function saveSettings(kv: KeyValueStore, s: Settings): Promise<void> {
  await kv.set(KEY, JSON.stringify(s));
}
