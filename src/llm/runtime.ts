import { initLlama } from 'llama.rn';
import type { InitFn } from '../core/llm/onDevice';
import { OnDeviceModel } from '../core/llm/onDevice';
import type { ModelSpec } from '../core/llm/models';
import { findModel } from '../core/llm/models';
import { isDownloaded, modelPath } from './modelStore';

/**
 * Holds the one on-device model instance. Rebuilt only when the chosen models change, and the old
 * one is released first: two 4B models never sit in memory together.
 */

let current: { key: string; model: OnDeviceModel } | null = null;
let loadListener: ((which: 'chat' | 'embed', progress: number) => void) | null = null;

export function onModelLoadProgress(fn: typeof loadListener): void {
  loadListener = fn;
}

export type RuntimeStatus =
  | { ok: true; model: OnDeviceModel; chat: ModelSpec; embed: ModelSpec }
  | { ok: false; reason: string };

export async function getRuntime(chatId: string, embedId: string): Promise<RuntimeStatus> {
  const chat = findModel(chatId);
  const embed = findModel(embedId);
  if (!chat || chat.role !== 'chat') return { ok: false, reason: 'Choose a chat model in Settings.' };
  if (!embed || embed.role !== 'embed') return { ok: false, reason: 'Choose an embedding model in Settings.' };
  const missing = [chat, embed].filter((m) => !isDownloaded(m));
  if (missing.length) {
    return { ok: false, reason: `Download ${missing.map((m) => m.name).join(' and ')} in Settings.` };
  }

  const key = `${chat.id}|${embed.id}`;
  if (current?.key !== key) {
    await current?.model.release();
    current = {
      key,
      model: new OnDeviceModel(
        {
          chatModelPath: modelPath(chat),
          embedModelPath: modelPath(embed),
          chatCtx: chat.nCtx,
          embedCtx: embed.nCtx,
          queryPrefix: embed.queryPrefix,
          documentPrefix: embed.documentPrefix,
        },
        initLlama as unknown as InitFn,
        (which, p) => loadListener?.(which, p),
      ),
    };
  }
  return { ok: true, model: current.model, chat, embed };
}

/**
 * Frees model memory (for example when the app goes to the background, where iOS kills
 * memory-heavy apps first). The instance stays; it reloads on next use.
 */
export async function releaseModelMemory(): Promise<void> {
  await current?.model.release();
}
