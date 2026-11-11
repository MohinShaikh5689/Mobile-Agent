import { chunkItem } from '../src/core/chunker';
import type { Embedder } from '../src/core/llm/types';
import type { LiveSearch } from '../src/core/retriever';
import { HybridRetriever } from '../src/core/retriever';
import type { Item, TimeWindow } from '../src/core/types';
import { InMemoryVectorIndex } from '../src/core/vectorIndex';
import type { Coverage } from '../src/core/watermark';
import { inWindow } from '../src/core/watermark';

/** Bag-of-words embedder over a fixed vocabulary: deterministic and good enough to test ranking. */
const VOCAB = ['visa', 'paperwork', 'pricing', 'vendor', 'lunch', 'contract', 'notice', 'invoice'];
const embedder: Embedder = {
  async embed(texts) {
    return texts.map((t) => {
      const words = t.toLowerCase().split(/\W+/);
      return VOCAB.map((v) => words.filter((w) => w === v).length);
    });
  },
};

const DAY = 86_400_000;
const T0 = new Date(2026, 9, 1).getTime();

const MAIL: Item[] = [
  { source: 'mail', id: 'm1', title: 'Visa paperwork', text: 'the visa paperwork is attached', date: T0 + 2 * DAY },
  { source: 'mail', id: 'm2', title: 'Lunch', text: 'lunch on friday?', date: T0 + 3 * DAY },
  { source: 'mail', id: 'm3', title: 'Vendor pricing', text: 'vendor pricing update for next year', date: T0 + 20 * DAY },
  { source: 'mail', id: 'm4', title: 'Visa update', text: 'visa appointment confirmed', date: T0 + 25 * DAY },
];

/** A fake Gmail: keyword OR match, within the window. */
function fakeLive(items: Item[], calls: TimeWindow[] = []): LiveSearch {
  return async ({ keywords, window }) => {
    calls.push({ ...window });
    return items.filter(
      (it) =>
        inWindow(it.date, window) &&
        (keywords.length === 0 || keywords.some((k) => `${it.title} ${it.text}`.toLowerCase().includes(k))),
    );
  };
}

async function indexUpTo(index: InMemoryVectorIndex, items: Item[], watermark: number) {
  for (const it of items.filter((i) => i.date < watermark)) {
    const chunks = chunkItem(it, it.text);
    const vecs = await embedder.embed(chunks.map((c) => c.text));
    await index.upsert(chunks.map((c, i) => ({ ...c, embedding: vecs[i] ?? [] })));
  }
}

function retrieverWith(index: InMemoryVectorIndex, coverage: Coverage | undefined, calls: TimeWindow[] = []) {
  return new HybridRetriever({
    index,
    embedder,
    live: { mail: fakeLive(MAIL, calls) },
    coverage: async () => (coverage ? { mail: coverage } : {}),
    k: 3,
  });
}

const request = (window: TimeWindow | null) => ({
  question: 'the thread about the visa paperwork',
  keywords: ['visa', 'paperwork'],
  window,
  sources: ['mail' as const],
});

describe('HybridRetriever', () => {
  it('answers an old window from the index alone', async () => {
    const index = new InMemoryVectorIndex();
    const wm = T0 + 10 * DAY;
    await indexUpTo(index, MAIL, wm);
    const calls: TimeWindow[] = [];
    const { results, trace } = await retrieverWith(index, { floor: T0, watermark: wm }, calls).retrieve(
      request({ start: T0, end: T0 + 5 * DAY }),
    );
    expect(results[0]?.id).toBe('m1');
    expect(results[0]?.origin).toBe('index');
    expect(calls).toEqual([]);
    expect(trace[0]?.liveWindows).toEqual([]);
  });

  it('merges index and live for an undated question', async () => {
    const index = new InMemoryVectorIndex();
    const wm = T0 + 10 * DAY;
    await indexUpTo(index, MAIL, wm);
    const { results } = await retrieverWith(index, { floor: T0, watermark: wm }).retrieve(request(null));
    const ids = results.map((r) => `${r.id}:${r.origin}`);
    expect(ids).toContain('m1:index');
    expect(ids).toContain('m4:live');
  });

  it('gives the same top answer whichever side of the watermark the item falls', async () => {
    // Build-plan check for phase 2: same question, same answer, before or after the watermark.
    const before = new InMemoryVectorIndex();
    await indexUpTo(before, MAIL, T0 + 30 * DAY); // m1 indexed
    const after = new InMemoryVectorIndex();
    await indexUpTo(after, MAIL, T0 + 1 * DAY); // nothing indexed; m1 is live

    const a = await retrieverWith(before, { floor: T0, watermark: T0 + 30 * DAY }).retrieve(request(null));
    const b = await retrieverWith(after, { floor: T0, watermark: T0 + 1 * DAY }).retrieve(request(null));
    expect(a.results[0]?.id).toBe('m1');
    expect(b.results[0]?.id).toBe('m1');
    expect(a.results[0]?.origin).toBe('index');
    expect(b.results[0]?.origin).toBe('live');
  });

  it('falls back to a live query when the index result is weak', async () => {
    const index = new InMemoryVectorIndex(); // covered range claims completeness, but holds nothing
    const calls: TimeWindow[] = [];
    const window = { start: T0, end: T0 + 5 * DAY };
    const { results, trace } = await retrieverWith(index, { floor: T0, watermark: T0 + 10 * DAY }, calls).retrieve(
      request(window),
    );
    expect(trace[0]?.fallback).toBe(true);
    expect(calls).toEqual([window]);
    expect(results[0]?.id).toBe('m1');
  });

  it('keeps index results when the live fetch fails', async () => {
    const index = new InMemoryVectorIndex();
    const wm = T0 + 10 * DAY;
    await indexUpTo(index, MAIL, wm);
    const r = new HybridRetriever({
      index,
      embedder,
      live: {
        mail: async () => {
          throw new Error('offline');
        },
      },
      coverage: async () => ({ mail: { floor: T0, watermark: wm } }),
    });
    const { results, trace } = await r.retrieve(request(null));
    expect(results[0]?.id).toBe('m1');
    expect(trace[0]?.error).toBe('offline');
  });
});
