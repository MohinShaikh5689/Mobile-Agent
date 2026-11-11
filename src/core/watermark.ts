import type { ScoredItem, TimeWindow } from './types';

/**
 * What the local index holds for one source: every item dated in [floor, watermark).
 * - watermark: the time up to which the index is known to be complete (the start of the last sync).
 * - floor: how far back the first sync reached. Missing = the index goes back to the beginning.
 */
export interface Coverage {
  floor?: number;
  watermark: number;
}

/**
 * The watermark rule (Retrieval design):
 * - A window that lies inside the covered range is answered from the index.
 * - A window that reaches past the watermark, or has no date, runs both: vector search for the
 *   covered part, a live fetch filtered to "since watermark" for the rest.
 * - The first sync only backfills a limited history, so anything before the floor is fetched live too.
 * - No coverage yet (nothing synced) means everything is fetched live.
 */
export interface RetrievalPlan {
  /** Window to search in the local index, or null to skip the index. */
  index: TimeWindow | null;
  /** Windows to fetch live from the API (0, 1 or 2 of them). */
  live: TimeWindow[];
}

export function planRetrieval(window: TimeWindow | null, coverage: Coverage | undefined): RetrievalPlan {
  const w: TimeWindow = window ?? {};
  if (!coverage) return { index: null, live: [{ ...w }] };

  const { floor, watermark } = coverage;
  const start = w.start;
  const end = w.end;

  // Overlap of the window with [floor, watermark).
  const idxStart = floor === undefined ? start : start === undefined ? floor : Math.max(start, floor);
  const idxEnd = end === undefined ? watermark : Math.min(end, watermark);
  const overlaps = idxStart === undefined || idxStart < idxEnd;
  const index: TimeWindow | null = overlaps ? { start: idxStart, end: idxEnd } : null;

  const live: TimeWindow[] = [];
  // Part before the floor.
  if (floor !== undefined && (start === undefined || start < floor)) {
    const e = end === undefined ? floor : Math.min(end, floor);
    if (start === undefined || start < e) live.push({ start, end: e });
  }
  // Part since the watermark.
  if (end === undefined || end > watermark) {
    const s = start === undefined ? watermark : Math.max(start, watermark);
    if (end === undefined || s < end) live.push({ start: s, end });
  }
  return { index, live };
}

/** Minimum best score for an index-only result to count as an answer rather than a miss. */
export const WEAK_SCORE = 0.35;

/**
 * An empty or weak index result for something that should exist triggers a live query instead of a
 * final "not found".
 */
export function isWeak(results: ScoredItem[], minScore = WEAK_SCORE): boolean {
  if (results.length === 0) return true;
  return Math.max(...results.map((r) => r.score)) < minScore;
}

/**
 * Merge index and live hits. The same item can show up from both sides (it arrived during a sync,
 * or the weak-result fallback re-fetched it); the live copy wins a tie because it is fresher. Several
 * chunks of one item keep only the best one. Sorted by score, ties broken by newer date.
 */
export function mergeResults(lists: ScoredItem[][], limit: number): ScoredItem[] {
  const best = new Map<string, ScoredItem>();
  for (const list of lists) {
    for (const r of list) {
      const key = `${r.source}:${r.id}`;
      const prev = best.get(key);
      if (
        !prev ||
        r.score > prev.score + 1e-9 ||
        (Math.abs(r.score - prev.score) <= 1e-9 && r.origin === 'live' && prev.origin === 'index')
      ) {
        best.set(key, r);
      }
    }
  }
  return [...best.values()].sort((a, b) => b.score - a.score || b.date - a.date).slice(0, limit);
}

export function inWindow(t: number, w: TimeWindow | null): boolean {
  if (!w) return true;
  if (w.start !== undefined && t < w.start) return false;
  if (w.end !== undefined && t >= w.end) return false;
  return true;
}
