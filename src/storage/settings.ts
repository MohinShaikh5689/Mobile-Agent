import type { ModelSpec } from '../core/llm/models';
import { DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, findModel } from '../core/llm/models';
import type { KeyValueStore } from '../core/services';

export interface Settings {
  /** Model ids from src/core/llm/models.ts or customModels. Both run on the phone. */
  chatModel: string;
  embedModel: string;
  /** Chat models the user added from a .gguf link. */
  customModels: ModelSpec[];
  /** Per-model download URL overrides, in case a file moves. */
  modelUrls: Record<string, string>;
  maxSteps: number;
}

export const DEFAULT_SETTINGS: Settings = {
  chatModel: DEFAULT_CHAT_MODEL,
  embedModel: DEFAULT_EMBED_MODEL,
  customModels: [],
  modelUrls: {},
  maxSteps: 5,
};

const KEY = 'settings';

/** `defaultChat`: the chat model to start with on a fresh install (the one recommended for this phone). */
export async function loadSettings(kv: KeyValueStore, defaultChat?: string): Promise<Settings> {
  const raw = await kv.get(KEY);
  if (!raw) return defaultChat ? { ...DEFAULT_SETTINGS, chatModel: defaultChat } : DEFAULT_SETTINGS;
  try {
    const parsed = JSON.parse(raw) as Partial<Settings>;
    const customModels = Array.isArray(parsed.customModels) ? parsed.customModels : [];
    // Ids that are not in the catalog (such as Ollama names saved by older builds) fall back to the defaults.
    const chat = findModel(parsed.chatModel ?? '', customModels);
    const embed = findModel(parsed.embedModel ?? '', customModels);
    return {
      chatModel: chat?.role === 'chat' ? chat.id : DEFAULT_SETTINGS.chatModel,
      embedModel: embed?.role === 'embed' ? embed.id : DEFAULT_SETTINGS.embedModel,
      customModels,
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
