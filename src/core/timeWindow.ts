import type { TimeWindow } from './types';

/**
 * Turns date phrases in a request into a time window, with plain rules rather than the model's
 * judgment (Retrieval design: "Routing uses a few keyword and date-pattern rules").
 * Returns null when the request names no time at all.
 * All dates are in the device's local time zone.
 */

const DAY = 24 * 60 * 60 * 1000;
const WEEKDAYS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
const MONTHS = [
  'january', 'february', 'march', 'april', 'may', 'june',
  'july', 'august', 'september', 'october', 'november', 'december',
];

function startOfDay(t: number): number {
  const d = new Date(t);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function addDays(dayStart: number, n: number): number {
  const d = new Date(dayStart);
  d.setDate(d.getDate() + n);
  return d.getTime();
}

/** Monday-based week start. */
function startOfWeek(t: number): number {
  const s = startOfDay(t);
  const dow = new Date(s).getDay(); // 0 = Sunday
  return addDays(s, -((dow + 6) % 7));
}

export interface ParsedWindow {
  window: TimeWindow;
  /** The phrase that matched, so it can be stripped from search keywords. */
  phrase: string;
}

export function parseTimeWindow(text: string, now: number): ParsedWindow | null {
  const q = text.toLowerCase();
  const today = startOfDay(now);
  let m: RegExpMatchArray | null;

  if ((m = q.match(/\btoday\b|\bthis morning\b|\btonight\b/))) {
    return { window: { start: today, end: addDays(today, 1) }, phrase: m[0] };
  }
  if ((m = q.match(/\byesterday\b/))) {
    return { window: { start: addDays(today, -1), end: today }, phrase: m[0] };
  }
  if ((m = q.match(/\btomorrow\b/))) {
    return { window: { start: addDays(today, 1), end: addDays(today, 2) }, phrase: m[0] };
  }
  if ((m = q.match(/\b(?:last|past)\s+(\d+)\s+(day|week|month)s?\b/))) {
    const n = Number(m[1]);
    const unit = m[2] === 'day' ? 1 : m[2] === 'week' ? 7 : 30;
    return { window: { start: addDays(today, -n * unit), end: now }, phrase: m[0] };
  }
  if ((m = q.match(/\bnext\s+(\d+)\s+(day|week)s?\b/))) {
    const n = Number(m[1]) * (m[2] === 'day' ? 1 : 7);
    return { window: { start: now, end: addDays(today, n + 1) }, phrase: m[0] };
  }
  if ((m = q.match(/\bthis week\b/))) {
    const s = startOfWeek(now);
    return { window: { start: s, end: addDays(s, 7) }, phrase: m[0] };
  }
  if ((m = q.match(/\blast week\b/))) {
    const s = startOfWeek(now);
    return { window: { start: addDays(s, -7), end: s }, phrase: m[0] };
  }
  if ((m = q.match(/\bnext week\b/))) {
    const s = startOfWeek(now);
    return { window: { start: addDays(s, 7), end: addDays(s, 14) }, phrase: m[0] };
  }
  if ((m = q.match(/\bthis month\b/))) {
    const d = new Date(today);
    const s = new Date(d.getFullYear(), d.getMonth(), 1).getTime();
    const e = new Date(d.getFullYear(), d.getMonth() + 1, 1).getTime();
    return { window: { start: s, end: e }, phrase: m[0] };
  }
  if ((m = q.match(/\blast month\b/))) {
    const d = new Date(today);
    const s = new Date(d.getFullYear(), d.getMonth() - 1, 1).getTime();
    const e = new Date(d.getFullYear(), d.getMonth(), 1).getTime();
    return { window: { start: s, end: e }, phrase: m[0] };
  }
  const wd = WEEKDAYS.join('|');
  if ((m = q.match(new RegExp(`\\b(last|this|next|on)?\\s*(${wd})\\b`)))) {
    const target = WEEKDAYS.indexOf(m[2] as string);
    const dow = new Date(today).getDay();
    const qualifier = m[1];
    let day: number;
    if (qualifier === 'next') {
      // "next friday" = the Friday of next week.
      day = addDays(startOfWeek(now), 7 + ((target + 6) % 7));
    } else if (qualifier === 'this') {
      day = addDays(startOfWeek(now), (target + 6) % 7);
    } else {
      // "last tuesday", "on tuesday", bare "tuesday": the most recent one before today.
      let diff = (dow - target + 7) % 7;
      if (diff === 0) diff = 7;
      day = addDays(today, -diff);
    }
    return { window: { start: day, end: addDays(day, 1) }, phrase: m[0].trim() };
  }
  const mo = MONTHS.join('|');
  if ((m = q.match(new RegExp(`\\b(?:in|during|from)\\s+(${mo})(?:\\s+(\\d{4}))?\\b`)))) {
    const month = MONTHS.indexOf(m[1] as string);
    const nowDate = new Date(today);
    let year = m[2] ? Number(m[2]) : nowDate.getFullYear();
    // "in March" said in February means last March.
    if (!m[2] && month > nowDate.getMonth()) year -= 1;
    const s = new Date(year, month, 1).getTime();
    const e = new Date(year, month + 1, 1).getTime();
    return { window: { start: s, end: e }, phrase: m[0] };
  }
  if ((m = q.match(/\b(?:in|during)\s+(\d{4})\b/))) {
    const year = Number(m[1]);
    return {
      window: { start: new Date(year, 0, 1).getTime(), end: new Date(year + 1, 0, 1).getTime() },
      phrase: m[0],
    };
  }
  if ((m = q.match(/\b(?:recent|recently|latest|lately)\b/))) {
    return { window: { start: addDays(today, -7), end: now }, phrase: m[0] };
  }
  return null;
}

export { DAY };
