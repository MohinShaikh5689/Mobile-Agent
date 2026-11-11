import { Directory, DownloadTask, File, Paths } from 'expo-file-system';
import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import type { ModelSpec } from '../core/llm/models';
import { formatBytes } from '../core/llm/models';

/**
 * Downloads GGUF model files into Documents/models. A download goes to `<file>.part` and is renamed
 * only when complete, so a half-written file is never loaded as a model.
 */

const KEEP_AWAKE_TAG = 'model-download';

function modelsDir(): Directory {
  const dir = new Directory(Paths.document, 'models');
  if (!dir.exists) dir.create({ intermediates: true, idempotent: true });
  return dir;
}

export function modelFile(spec: ModelSpec): File {
  return new File(modelsDir(), spec.fileName);
}

/** Native path for llama.rn (it takes a plain path, not a file:// URI). */
export function modelPath(spec: ModelSpec): string {
  return modelFile(spec).uri.replace(/^file:\/\//, '');
}

export function isDownloaded(spec: ModelSpec): boolean {
  const f = modelFile(spec);
  return f.exists && (f.size ?? 0) > 1024 * 1024;
}

/** Size of the downloaded file in bytes, or 0. */
export function downloadedSize(spec: ModelSpec): number {
  const f = modelFile(spec);
  return f.exists ? (f.size ?? 0) : 0;
}

export function deleteModel(spec: ModelSpec): void {
  for (const f of [modelFile(spec), new File(modelsDir(), `${spec.fileName}.part`)]) {
    if (f.exists) f.delete();
  }
}

export interface DownloadHandle {
  done: Promise<void>;
  cancel(): void;
}

/** Turns a native download error into something the user can act on. */
export function explainDownloadError(message: string, url: string): string {
  const host = url.replace(/^https?:\/\//, '').split('/')[0] ?? url;
  const status = message.match(/HTTP (\d{3})|status (\d{3})/);
  const code = status ? Number(status[1] ?? status[2]) : null;
  if (code === 404) return `The file was not found at that link (HTTP 404). Use Edit URL to paste a working .gguf link.`;
  if (code === 401 || code === 403) {
    return `${host} refused the download (HTTP ${code}). The model may be gated: pick a different link with Edit URL.`;
  }
  if (code) return `${host} returned HTTP ${code}. Try again, or use Edit URL.`;
  if (/offline|internet|network|timed out|host/i.test(message)) return `Could not reach ${host}: ${message}`;
  return message;
}

export function downloadModel(spec: ModelSpec, url: string, onProgress: (fraction: number) => void): DownloadHandle {
  const free = Paths.availableDiskSpace;
  // Leave headroom for the index and the OS.
  if (free > 0 && free < spec.sizeBytes + 500 * 1024 * 1024) {
    return {
      done: Promise.reject(new Error(`Not enough free space: ${formatBytes(spec.sizeBytes)} needed, ${formatBytes(free)} free.`)),
      cancel: () => undefined,
    };
  }

  const part = new File(modelsDir(), `${spec.fileName}.part`);
  if (part.exists) part.delete();
  const task = new DownloadTask(url, part, {
    // Foreground: the app stays open with the screen awake while downloading. iOS background
    // sessions can be deferred or throttled, which looks like a download that never starts.
    sessionType: 'foreground',
    onProgress: ({ bytesWritten, totalBytes }) => {
      const total = totalBytes > 0 ? totalBytes : spec.sizeBytes;
      if (total > 0) onProgress(Math.min(1, bytesWritten / total));
    },
  });

  const done = (async () => {
    await activateKeepAwakeAsync(KEEP_AWAKE_TAG);
    try {
      let file: File | null;
      try {
        file = await task.downloadAsync();
      } catch (e) {
        throw new Error(explainDownloadError((e as Error).message ?? String(e), url));
      }
      if (!file) throw new Error('Download was cancelled.');
      if ((file.size ?? 0) < 1024 * 1024) {
        const head = file.exists ? (await file.text()).replace(/\s+/g, ' ').slice(0, 160) : '';
        file.delete();
        throw new Error(`The link did not return a model file${head ? `: "${head}"` : ''}. Use Edit URL to fix it.`);
      }
      const target = modelFile(spec);
      if (target.exists) target.delete();
      await file.move(target);
      onProgress(1);
    } finally {
      deactivateKeepAwake(KEEP_AWAKE_TAG);
      task.release();
    }
  })();

  return { done, cancel: () => task.cancel() };
}
