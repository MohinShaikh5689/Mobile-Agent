/**
 * Resumable downloads for multi-gigabyte model files.
 *
 * The file is fetched in fixed-size pieces with HTTP Range requests and each piece is appended
 * to a `.part` file. A dropped connection costs one piece, not the whole download: the piece is
 * retried with backoff, and a later "Resume" (even after the app was closed) continues from the
 * size of the `.part` file. Servers that ignore Range get a single whole-file download instead.
 *
 * The I/O is injected, so this runs in tests without a phone.
 */

export type DownloadPhase = 'connecting' | 'downloading' | 'retrying' | 'finishing' | 'paused' | 'failed' | 'done';

export interface DownloadStatus {
  phase: DownloadPhase;
  /** Bytes on disk so far. */
  bytes: number;
  /** Total size, when the server reported it. */
  total: number | null;
  /** Recent speed in bytes per second (0 until measured). */
  speed: number;
  /** Seconds left at the recent speed, when known. */
  eta: number | null;
  /** Retry attempt for the current piece (1-based), while retrying. */
  attempt?: number;
  maxAttempts?: number;
  /** Seconds until the next retry, while retrying. */
  retryIn?: number;
  message?: string;
}

export class RangeUnsupportedError extends Error {}
/** The server says the requested range starts past the end: the file is complete. */
export class RangeNotSatisfiableError extends Error {}
export class DownloadCancelled extends Error {}

export interface DownloadIO {
  /** Size reported by the server (following redirects), or null if unknown. */
  probeSize(): Promise<number | null>;
  /** Bytes already in the .part file. */
  partSize(): Promise<number>;
  /**
   * Fetches bytes [start, end] (inclusive) and appends them to the .part file. Reports bytes of
   * this piece as they arrive. Throws RangeUnsupportedError if the server ignored the Range, and
   * RangeNotSatisfiableError on HTTP 416. Returns the number of bytes appended.
   */
  fetchPiece(start: number, end: number, onBytes: (n: number) => void): Promise<number>;
  /** Single whole-file download into the .part file (for servers without Range support). */
  fetchWhole(onBytes: (n: number) => void): Promise<void>;
  /** Renames .part to the final file. */
  finish(): Promise<void>;
  sleep(ms: number): Promise<void>;
  now(): number;
}

export interface DownloadOptions {
  pieceSize?: number;
  /** Attempts per piece before giving up (the progress stays on disk for a later Resume). */
  maxAttempts?: number;
  /** Backoff before attempt n+1, in ms. */
  backoff?: (attempt: number) => number;
}

export const PIECE_SIZE = 32 * 1024 * 1024;
const defaultBackoff = (attempt: number) => Math.min(30_000, 2000 * 2 ** (attempt - 1));

/** Tracks speed over a sliding window of recent samples. */
class SpeedMeter {
  private samples: { t: number; bytes: number }[] = [];
  constructor(private windowMs = 5000) {}
  add(t: number, bytes: number): number {
    this.samples.push({ t, bytes });
    while (this.samples.length > 2 && t - (this.samples[0] as { t: number }).t > this.windowMs) this.samples.shift();
    const first = this.samples[0] as { t: number; bytes: number };
    const dt = (t - first.t) / 1000;
    return dt > 0.5 ? Math.max(0, (bytes - first.bytes) / dt) : 0;
  }
  reset() {
    this.samples = [];
  }
}

export class ResumableDownload {
  private cancelled = false;
  private meter = new SpeedMeter();
  private lastEmit = 0;

  constructor(
    private io: DownloadIO,
    private onStatus: (s: DownloadStatus) => void,
    private opts: DownloadOptions = {},
  ) {}

  /** Stops after the current piece; what is on disk is kept for Resume. */
  cancel(): void {
    this.cancelled = true;
  }

  private emit(s: DownloadStatus, force = false) {
    const t = this.io.now();
    // At most ~4 updates a second, except phase changes.
    if (!force && t - this.lastEmit < 250) return;
    this.lastEmit = t;
    this.onStatus(s);
  }

  private progress(bytes: number, total: number | null, force = false) {
    const speed = this.meter.add(this.io.now(), bytes);
    const eta = total && speed > 0 ? Math.max(0, (total - bytes) / speed) : null;
    this.emit({ phase: 'downloading', bytes, total, speed, eta }, force);
  }

  async run(): Promise<void> {
    const pieceSize = this.opts.pieceSize ?? PIECE_SIZE;
    const maxAttempts = this.opts.maxAttempts ?? 6;
    const backoff = this.opts.backoff ?? defaultBackoff;

    let have = await this.io.partSize();
    this.emit({ phase: 'connecting', bytes: have, total: null, speed: 0, eta: null }, true);
    let total: number | null = null;
    try {
      total = await this.io.probeSize();
    } catch {
      total = null; // a HEAD failure is not fatal: pieces will find the end
    }

    let attempt = 0;
    while (total === null || have < total) {
      if (this.cancelled) {
        this.emit({ phase: 'paused', bytes: have, total, speed: 0, eta: null }, true);
        throw new DownloadCancelled('Paused.');
      }
      const end = total === null ? have + pieceSize - 1 : Math.min(total, have + pieceSize) - 1;
      try {
        const base = have;
        const got = await this.io.fetchPiece(base, end, (n) => this.progress(base + n, total));
        have += got;
        attempt = 0;
        this.progress(have, total, true);
        // Unknown size: a short piece means the end of the file.
        if (total === null && got < end - base + 1) break;
      } catch (e) {
        if (e instanceof RangeNotSatisfiableError) break; // nothing left to fetch
        if (e instanceof RangeUnsupportedError) {
          if (have > 0) throw new Error('The server stopped supporting resumable downloads. Delete the partial file and try again.');
          await this.io.fetchWhole((n) => this.progress(n, total));
          have = await this.io.partSize();
          break;
        }
        if (this.cancelled) continue;
        attempt++;
        // Re-read what is on disk: a piece may have been partly appended.
        have = await this.io.partSize();
        if (attempt >= maxAttempts) {
          const message = `${(e as Error).message} (stopped after ${maxAttempts} tries; tap Resume to continue from ${formatSize(have)})`;
          this.emit({ phase: 'failed', bytes: have, total, speed: 0, eta: null, message }, true);
          throw new Error(message);
        }
        const wait = backoff(attempt);
        this.meter.reset();
        for (let left = Math.ceil(wait / 1000); left > 0 && !this.cancelled; left--) {
          this.emit(
            { phase: 'retrying', bytes: have, total, speed: 0, eta: null, attempt: attempt + 1, maxAttempts, retryIn: left, message: (e as Error).message },
            true,
          );
          await this.io.sleep(Math.min(1000, wait));
        }
      }
    }

    this.emit({ phase: 'finishing', bytes: have, total, speed: 0, eta: null }, true);
    await this.io.finish();
    this.emit({ phase: 'done', bytes: have, total: total ?? have, speed: 0, eta: 0 }, true);
  }
}

export function formatSize(n: number): string {
  const GB = 1024 ** 3;
  const MB = 1024 ** 2;
  if (n >= GB) return `${(n / GB).toFixed(2)} GB`;
  if (n >= MB) return `${(n / MB).toFixed(0)} MB`;
  return `${Math.max(0, Math.round(n / 1024))} KB`;
}

export function formatEta(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return '';
  if (seconds < 60) return 'under a minute left';
  if (seconds < 3600) return `about ${Math.round(seconds / 60)} min left`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `about ${h} h ${m} min left`;
}

/** One line for the UI. */
export function describeStatus(s: DownloadStatus): string {
  const of = s.total ? `${formatSize(s.bytes)} of ${formatSize(s.total)}` : formatSize(s.bytes);
  const pct = s.total ? ` (${Math.floor((s.bytes / s.total) * 100)}%)` : '';
  switch (s.phase) {
    case 'connecting':
      return s.bytes > 0 ? `Resuming from ${formatSize(s.bytes)}…` : 'Connecting…';
    case 'downloading': {
      const speed = s.speed > 0 ? ` · ${formatSize(s.speed)}/s` : '';
      const eta = formatEta(s.eta);
      return `${of}${pct}${speed}${eta ? ` · ${eta}` : ''}`;
    }
    case 'retrying':
      return `Connection problem, retrying in ${s.retryIn ?? 0} s (try ${s.attempt} of ${s.maxAttempts}) · ${of} kept`;
    case 'finishing':
      return 'Finishing…';
    case 'paused':
      return `Paused at ${of}${pct}`;
    case 'failed':
      return s.message ?? 'Download failed.';
    case 'done':
      return 'Downloaded';
  }
}

/** GGUF files start with the ASCII bytes "GGUF". */
export function isGguf(header: Uint8Array): boolean {
  return header.length >= 4 && header[0] === 0x47 && header[1] === 0x47 && header[2] === 0x55 && header[3] === 0x46;
}
