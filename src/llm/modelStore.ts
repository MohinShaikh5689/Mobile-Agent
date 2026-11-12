import { Directory, DownloadTask, File, FileMode, Paths } from 'expo-file-system';
import type { DownloadIO, DownloadStatus } from '../core/download';
import { isGguf, RangeNotSatisfiableError, RangeUnsupportedError, ResumableDownload } from '../core/download';
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

/** Turns a native download error into something the user can act on. */
export function explainDownloadError(message: string, url: string): string {
  const host = url.replace(/^https?:\/\//, '').split('/')[0] ?? url;
  const status = message.match(/HTTP (\d{3})|status (\d{3})/);
  const code = status ? Number(status[1] ?? status[2]) : null;
  if (code === 404) return `The file was not found at that link (HTTP 404). Use Edit URL to paste a working .gguf link.`;
  if (code === 401 || code === 403) {
    return `${host} refused the download (HTTP ${code}). The model may be gated: pick a different link with Edit URL.`;
  }
  if (code) return `${host} returned HTTP ${code}.`;
  if (/offline|internet|network|timed out|host|connection/i.test(message)) return `Could not reach ${host}: ${message}`;
  return message;
}

export interface DownloadHandle {
  done: Promise<void>;
  /** Stops the download; what is downloaded so far stays on disk for Resume. */
  pause(): void;
}

const partFile = (spec: ModelSpec) => new File(modelsDir(), `${spec.fileName}.part`);
const pieceFile = (spec: ModelSpec) => new File(modelsDir(), `${spec.fileName}.piece`);
const metaFile = (spec: ModelSpec) => new File(modelsDir(), `${spec.fileName}.part.json`);

/** Bytes of an unfinished download kept on disk (0 when there is none). */
export function partialBytes(spec: ModelSpec): number {
  const f = partFile(spec);
  return f.exists ? (f.size ?? 0) : 0;
}

/** Drops an unfinished download. */
export function discardPartial(spec: ModelSpec): void {
  for (const f of [partFile(spec), pieceFile(spec), metaFile(spec)]) if (f.exists) f.delete();
}

async function readHeader(file: File): Promise<Uint8Array> {
  const h = file.open(FileMode.ReadOnly);
  try {
    return await h.readBytes(4);
  } finally {
    h.close();
  }
}

/** Server-reported size, following redirects. Hugging Face also sends x-linked-size. */
async function probeSize(url: string): Promise<number | null> {
  const res = await fetch(url, { method: 'HEAD' });
  if (!res.ok) return null;
  const n = Number(res.headers.get('x-linked-size') ?? res.headers.get('content-length'));
  return Number.isFinite(n) && n > 1024 * 1024 ? n : null;
}

/**
 * Downloads a model in 32 MB pieces with resume and retries (src/core/download.ts). Progress goes
 * to `onStatus`; a paused or failed download keeps its bytes on disk and continues on the next call.
 */
export function downloadModel(spec: ModelSpec, url: string, onStatus: (s: DownloadStatus) => void): DownloadHandle {
  const part = partFile(spec);
  const piece = pieceFile(spec);
  const meta = metaFile(spec);
  let current: DownloadTask | null = null;

  // A different link means a different file: start over.
  if (part.exists && (!meta.exists || meta.textSync() !== url)) discardPartial(spec);

  const io: DownloadIO = {
    probeSize: () => probeSize(url),
    partSize: async () => (part.exists ? (part.size ?? 0) : 0),
    async fetchPiece(start, end, onBytes) {
      if (piece.exists) piece.delete();
      const expected = end - start + 1;
      let ignoredRange = false;
      const task = new DownloadTask(url, piece, {
        sessionType: 'foreground',
        headers: { Range: `bytes=${start}-${end}` },
        onProgress: ({ bytesWritten }) => {
          // A server that ignores Range sends the whole file: stop at once instead of fetching gigabytes.
          if (bytesWritten > expected && !ignoredRange) {
            ignoredRange = true;
            task.cancel();
          }
          onBytes(Math.min(bytesWritten, expected));
        },
      });
      current = task;
      try {
        let file: File | null;
        try {
          file = await task.downloadAsync();
        } catch (e) {
          if (ignoredRange) throw new RangeUnsupportedError('Range ignored');
          const raw = (e as Error).message ?? String(e);
          if (/\b416\b/.test(raw)) throw new RangeNotSatisfiableError(raw);
          throw new Error(explainDownloadError(raw, url));
        }
        if (ignoredRange) throw new RangeUnsupportedError('Range ignored');
        if (!file) throw new Error('Stopped.');
        const size = file.size ?? 0;
        if (size > expected) throw new RangeUnsupportedError('Range ignored');
        // Append the piece to the .part file.
        if (!part.exists) {
          part.create();
          meta.write(url);
        }
        const bytes = await file.bytes();
        const h = part.open(FileMode.ReadWrite);
        try {
          h.offset = h.size ?? 0;
          await h.writeBytes(bytes);
        } finally {
          h.close();
        }
        file.delete();
        return size;
      } finally {
        current = null;
        task.release();
      }
    },
    async fetchWhole(onBytes) {
      if (part.exists) part.delete();
      const task = new DownloadTask(url, part, {
        sessionType: 'foreground',
        onProgress: ({ bytesWritten }) => onBytes(bytesWritten),
      });
      current = task;
      try {
        const file = await task.downloadAsync().catch((e: Error) => {
          throw new Error(explainDownloadError(e.message ?? String(e), url));
        });
        if (!file) throw new Error('Stopped.');
        meta.write(url);
      } finally {
        current = null;
        task.release();
      }
    },
    async finish() {
      if (!isGguf(await readHeader(part))) {
        const head = (await part.text()).replace(/\s+/g, ' ').slice(0, 120);
        discardPartial(spec);
        throw new Error(`The link did not return a model file${head ? `: "${head}"` : ''}. Use Edit URL to fix it.`);
      }
      const target = modelFile(spec);
      if (target.exists) target.delete();
      await part.move(target);
      if (meta.exists) meta.delete();
      if (piece.exists) piece.delete();
    },
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
    now: () => Date.now(),
  };

  const job = new ResumableDownload(io, onStatus);
  const done = (async () => {
    const free = Paths.availableDiskSpace;
    const need = Math.max(0, spec.sizeBytes - partialBytes(spec)) + 500 * 1024 * 1024;
    if (free > 0 && spec.sizeBytes > 0 && free < need) {
      throw new Error(`Not enough free space: about ${formatBytes(need)} needed, ${formatBytes(free)} free.`);
    }
    await activateKeepAwakeAsync(KEEP_AWAKE_TAG);
    try {
      await job.run();
    } finally {
      deactivateKeepAwake(KEEP_AWAKE_TAG);
    }
  })();

  return {
    done,
    pause: () => {
      job.cancel();
      current?.cancel();
    },
  };
}

/** Opens the Files picker for a .gguf model. Returns null when cancelled. */
export async function pickModelFile(): Promise<File | null> {
  const res = await File.pickFileAsync({});
  return res.canceled ? null : res.result;
}

/**
 * Moves a picked .gguf file into the models folder as `spec`'s file. The picker hands over a
 * temporary copy, so moving it costs no extra space.
 */
export async function importModelFile(spec: ModelSpec, picked: File): Promise<void> {
  if (!picked.exists) throw new Error('The selected file is no longer available.');
  if (!isGguf(await readHeader(picked))) {
    throw new Error(`${picked.name} is not a GGUF model file. Pick a .gguf file (for example a Q4_K_M quantization).`);
  }
  discardPartial(spec);
  const target = modelFile(spec);
  if (target.exists) target.delete();
  try {
    await picked.move(target);
  } catch {
    await picked.copy(target);
  }
}
