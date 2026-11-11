import type { KeyValueStore } from '../core/services';

export interface Settings {
  /** Ollama on the laptop, reachable over Wi-Fi. Run it with OLLAMA_HOST=0.0.0.0. */
  ollamaUrl: string;
  chatModel: string;
  embedModel: string;
  maxSteps: number;
}

export const DEFAULT_SETTINGS: Settings = {
  ollamaUrl: 'http://192.168.1.10:11434',
  chatModel: 'qwen3:4b-instruct',
  embedModel: 'nomic-embed-text',
  maxSteps: 5,
};

const KEY = 'settings';

export async function loadSettings(kv: KeyValueStore): Promise<Settings> {
  const raw = await kv.get(KEY);
  if (!raw) return DEFAULT_SETTINGS;
  try {
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export async function saveSettings(kv: KeyValueStore, s: Settings): Promise<void> {
  await kv.set(KEY, JSON.stringify(s));
}
