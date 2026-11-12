import type { DownloadIO, DownloadStatus } from '../src/core/download';
import {
  describeStatus,
  DownloadCancelled,
  isGguf,
  RangeNotSatisfiableError,
  RangeUnsupportedError,
  ResumableDownload,
} from '../src/core/download';
import { customChatModel, importedChatModel } from '../src/core/llm/models';

const MB = 1024 * 1024;

/** A fake server + disk: `failOn` makes given piece starts throw a network error N times. */
function fakeIO(opts: {
  total: number | null;
  realSize?: number;
  have?: number;
  failOn?: Record<number, number>;
  noRange?: boolean;
  onPiece?: (start: number) => void;
}) {
  const realSize = opts.realSize ?? opts.total ?? 0;
  const state = { part: opts.have ?? 0, finished: false, pieces: [] as [number, number][], whole: 0, slept: 0 };
  const failures = { ...(opts.failOn ?? {}) };
  let t = 0;
  const io: DownloadIO = {
    probeSize: async () => opts.total,
    partSize: async () => state.part,
    async fetchPiece(start, end, onBytes) {
      opts.onPiece?.(start);
      if (opts.noRange) throw new RangeUnsupportedError('ignored');
      if (start >= realSize) throw new RangeNotSatisfiableError('HTTP 416');
      if ((failures[start] ?? 0) > 0) {
        failures[start] = (failures[start] ?? 0) - 1;
        onBytes(1000); // some bytes arrived before the drop, but the piece is not appended
        throw new Error('The network connection was lost.');
      }
      const n = Math.min(end, realSize - 1) - start + 1;
      onBytes(n);
      state.pieces.push([start, end]);
      state.part += n;
      return n;
    },
    async fetchWhole(onBytes) {
      state.whole++;
      onBytes(realSize);
      state.part = realSize;
    },
    async finish() {
      state.finished = true;
    },
    sleep: async (ms) => {
      state.slept += ms;
      t += ms;
    },
    now: () => (t += 100),
  };
  return { io, state };
}

const quick = { pieceSize: 10 * MB, maxAttempts: 3, backoff: () => 1000 };

describe('ResumableDownload', () => {
  it('downloads in pieces up to the reported size', async () => {
    const { io, state } = fakeIO({ total: 25 * MB });
    const seen: DownloadStatus[] = [];
    await new ResumableDownload(io, (s) => seen.push(s), quick).run();
    expect(state.pieces).toEqual([
      [0, 10 * MB - 1],
      [10 * MB, 20 * MB - 1],
      [20 * MB, 25 * MB - 1],
    ]);
    expect(state.finished).toBe(true);
    expect(seen[0]?.phase).toBe('connecting');
    expect(seen.at(-1)).toMatchObject({ phase: 'done', bytes: 25 * MB });
  });

  it('retries a dropped piece and keeps what was already downloaded', async () => {
    const { io, state } = fakeIO({ total: 30 * MB, failOn: { [10 * MB]: 2 } });
    const seen: DownloadStatus[] = [];
    await new ResumableDownload(io, (s) => seen.push(s), quick).run();
    expect(state.part).toBe(30 * MB);
    // Only the failed piece was fetched again; the first piece was not.
    expect(state.pieces.filter(([s]) => s === 0)).toHaveLength(1);
    const retry = seen.find((s) => s.phase === 'retrying');
    expect(retry).toMatchObject({ bytes: 10 * MB, attempt: 2, maxAttempts: 3 });
    expect(describeStatus(retry as DownloadStatus)).toMatch(/retrying in \d+ s \(try 2 of 3\) · 10 MB of 30 MB kept/);
  });

  it('stops after the last attempt and resumes later from the partial file', async () => {
    const first = fakeIO({ total: 30 * MB, failOn: { [20 * MB]: 5 } });
    const seen: DownloadStatus[] = [];
    await expect(new ResumableDownload(first.io, (s) => seen.push(s), quick).run()).rejects.toThrow(/tap Resume to continue from 20 MB/);
    expect(seen.at(-1)?.phase).toBe('failed');

    // Later (even after an app restart): starts from 20 MB, fetches only the rest.
    const later = fakeIO({ total: 30 * MB, have: 20 * MB });
    await new ResumableDownload(later.io, () => undefined, quick).run();
    expect(later.state.pieces).toEqual([[20 * MB, 30 * MB - 1]]);
    expect(later.state.finished).toBe(true);
  });

  it('falls back to one whole-file download when the server ignores Range', async () => {
    const { io, state } = fakeIO({ total: 25 * MB, noRange: true });
    await new ResumableDownload(io, () => undefined, quick).run();
    expect(state.whole).toBe(1);
    expect(state.finished).toBe(true);
  });

  it('finds the end without a reported size (short piece or HTTP 416)', async () => {
    const short = fakeIO({ total: null, realSize: 15 * MB });
    await new ResumableDownload(short.io, () => undefined, quick).run();
    expect(short.state.part).toBe(15 * MB);

    const exact = fakeIO({ total: null, realSize: 20 * MB });
    await new ResumableDownload(exact.io, () => undefined, quick).run();
    expect(exact.state.part).toBe(20 * MB);
    expect(exact.state.finished).toBe(true);
  });

  it('pauses on request and keeps the bytes', async () => {
    let job: ResumableDownload | undefined;
    const { io, state } = fakeIO({ total: 30 * MB, onPiece: (start) => start === 10 * MB && job?.cancel() });
    const seen: DownloadStatus[] = [];
    job = new ResumableDownload(io, (s) => seen.push(s), quick);
    await expect(job.run()).rejects.toBeInstanceOf(DownloadCancelled);
    expect(state.part).toBe(20 * MB); // the piece in flight completed
    expect(state.finished).toBe(false);
    expect(describeStatus(seen.at(-1) as DownloadStatus)).toBe('Paused at 20 MB of 30 MB (66%)');
  });
});

describe('download status text', () => {
  it('shows bytes, percent, speed and time left', () => {
    expect(
      describeStatus({ phase: 'downloading', bytes: 1.2 * 1024 ** 3, total: 2.5 * 1024 ** 3, speed: 4.3 * MB, eta: 330 }),
    ).toBe('1.20 GB of 2.50 GB (48%) · 4 MB/s · about 6 min left');
    expect(describeStatus({ phase: 'connecting', bytes: 0, total: null, speed: 0, eta: null })).toBe('Connecting…');
    expect(describeStatus({ phase: 'connecting', bytes: 300 * MB, total: null, speed: 0, eta: null })).toBe('Resuming from 300 MB…');
  });
});

describe('import', () => {
  it('recognises GGUF files by their header', () => {
    expect(isGguf(new TextEncoder().encode('GGUF\u0003\u0000'))).toBe(true);
    expect(isGguf(new TextEncoder().encode('<!DOCTYPE html>'))).toBe(false);
    expect(isGguf(new Uint8Array(2))).toBe(false);
  });

  it('makes an imported file a chat model with the same id a link would get', () => {
    const imported = importedChatModel('Llama-3.2-3B-Q4_K_M.gguf');
    expect(imported).toMatchObject({ role: 'chat', url: '', custom: true, fileName: 'Llama-3.2-3B-Q4_K_M.gguf' });
    expect(imported.id).toBe(customChatModel('', 'https://x.test/Llama-3.2-3B-Q4_K_M.gguf').id);
    expect(() => importedChatModel('notes.txt')).toThrow(/gguf/);
  });
});
