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

export function deleteModel(spec: ModelSpec): void {
  for (const f of [modelFile(spec), new File(modelsDir(), `${spec.fileName}.part`)]) {
    if (f.exists) f.delete();
  }
}

export interface DownloadHandle {
  done: Promise<void>;
  cancel(): void;
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
    onProgress: ({ bytesWritten, totalBytes }) => {
      const total = totalBytes > 0 ? totalBytes : spec.sizeBytes;
      onProgress(Math.min(1, bytesWritten / total));
    },
  });

  const done = (async () => {
    await activateKeepAwakeAsync(KEEP_AWAKE_TAG);
    try {
      const file = await task.downloadAsync();
      if (!file) throw new Error('Download was cancelled.');
      if ((file.size ?? 0) < 1024 * 1024) {
        const head = file.exists ? (await file.text()).slice(0, 200) : '';
        file.delete();
        throw new Error(`The server did not return a model file. ${head}`.trim());
      }
      const target = modelFile(spec);
      if (target.exists) target.delete();
      file.move(target);
      onProgress(1);
    } finally {
      deactivateKeepAwake(KEEP_AWAKE_TAG);
      task.release();
    }
  })();

  return { done, cancel: () => task.cancel() };
}
