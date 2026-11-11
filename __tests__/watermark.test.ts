import type { ScoredItem } from '../src/core/types';
import { isWeak, mergeResults, planRetrieval } from '../src/core/watermark';

const FLOOR = 1_000;
const WM = 5_000;
const cov = { floor: FLOOR, watermark: WM };

describe('planRetrieval', () => {
  it('fetches everything live when nothing is synced', () => {
    expect(planRetrieval({ start: 100, end: 200 }, undefined)).toEqual({ index: null, live: [{ start: 100, end: 200 }] });
  });

  it('uses only the index for a window inside the covered range', () => {
    expect(planRetrieval({ start: 2_000, end: 3_000 }, cov)).toEqual({ index: { start: 2_000, end: 3_000 }, live: [] });
  });

  it('treats a window ending exactly at the watermark as covered', () => {
    expect(planRetrieval({ start: 2_000, end: WM }, cov).live).toEqual([]);
  });

  it('uses only live for a window after the watermark', () => {
    expect(planRetrieval({ start: 6_000, end: 7_000 }, cov)).toEqual({ index: null, live: [{ start: 6_000, end: 7_000 }] });
  });

  it('splits a window that straddles the watermark', () => {
    expect(planRetrieval({ start: 4_000, end: 6_000 }, cov)).toEqual({
      index: { start: 4_000, end: WM },
      live: [{ start: WM, end: 6_000 }],
    });
  });

  it('runs both for a question with no date', () => {
    expect(planRetrieval(null, cov)).toEqual({
      index: { start: FLOOR, end: WM },
      live: [{ start: undefined, end: FLOOR }, { start: WM, end: undefined }],
    });
  });

  it('fetches the part before the floor live', () => {
    expect(planRetrieval({ start: 500, end: 2_000 }, cov)).toEqual({
      index: { start: FLOOR, end: 2_000 },
      live: [{ start: 500, end: FLOOR }],
    });
  });

  it('fetches a window entirely before the floor live', () => {
    expect(planRetrieval({ start: 100, end: 500 }, cov)).toEqual({ index: null, live: [{ start: 100, end: 500 }] });
  });

  it('has no floor gap when the index goes back to the beginning', () => {
    expect(planRetrieval(null, { watermark: WM })).toEqual({
      index: { start: undefined, end: WM },
      live: [{ start: WM, end: undefined }],
    });
  });
});

const hit = (id: string, score: number, origin: 'index' | 'live', date = 0): ScoredItem => ({
  source: 'mail',
  id,
  title: id,
  text: id,
  date,
  score,
  origin,
});

describe('mergeResults', () => {
  it('keeps one entry per item, the best score, and prefers live on a tie', () => {
    const merged = mergeResults(
      [
        [hit('a', 0.9, 'index'), hit('a', 0.5, 'index'), hit('b', 0.7, 'index')],
        [hit('b', 0.7, 'live'), hit('c', 0.8, 'live')],
      ],
      10,
    );
    expect(merged.map((m) => [m.id, m.origin])).toEqual([
      ['a', 'index'],
      ['c', 'live'],
      ['b', 'live'],
    ]);
  });

  it('respects the limit', () => {
    expect(mergeResults([[hit('a', 0.1, 'index'), hit('b', 0.2, 'index')]], 1).map((m) => m.id)).toEqual(['b']);
  });
});

describe('isWeak', () => {
  it('flags empty and low-scoring results', () => {
    expect(isWeak([])).toBe(true);
    expect(isWeak([hit('a', 0.2, 'index')])).toBe(true);
    expect(isWeak([hit('a', 0.2, 'index'), hit('b', 0.6, 'index')])).toBe(false);
  });
});
