import type { CalendarService } from '../core/services';
import type { Item, TimeWindow } from '../core/types';
import type { TokenProvider } from './http';
import { googleFetch, qs } from './http';

const BASE = 'https://www.googleapis.com/calendar/v3/calendars/primary/events';

interface CalEvent {
  id: string;
  summary?: string;
  description?: string;
  location?: string;
  htmlLink?: string;
  status?: string;
  start?: { dateTime?: string; date?: string };
  end?: { dateTime?: string; date?: string };
  organizer?: { email?: string; displayName?: string };
  attendees?: { email?: string; displayName?: string; responseStatus?: string; self?: boolean }[];
}

function when(t?: { dateTime?: string; date?: string }): number {
  if (t?.dateTime) return Date.parse(t.dateTime);
  if (t?.date) {
    const [y, m, d] = t.date.split('-').map(Number);
    return new Date(y ?? 1970, (m ?? 1) - 1, d ?? 1).getTime();
  }
  return 0;
}

function timeRange(e: CalEvent): string {
  if (e.start?.date && !e.start.dateTime) return 'all day';
  const fmt = (t: number) => {
    const d = new Date(t);
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  };
  return `${fmt(when(e.start))}–${fmt(when(e.end))}`;
}

function toItem(e: CalEvent, full: boolean): Item {
  const parts = [timeRange(e)];
  if (e.location) parts.push(`at ${e.location}`);
  const others = (e.attendees ?? []).filter((a) => !a.self);
  if (others.length) {
    const names = others.map((a) => a.displayName || a.email).filter(Boolean);
    parts.push(`with ${full ? names.join(', ') : names.slice(0, 4).join(', ') + (names.length > 4 ? ` +${names.length - 4}` : '')}`);
  }
  const mine = e.attendees?.find((a) => a.self)?.responseStatus;
  if (mine && mine !== 'accepted') parts.push(`(you: ${mine})`);
  if (full && e.description) parts.push(`\n${e.description.slice(0, 2000)}`);
  return {
    source: 'calendar',
    id: e.id,
    title: e.summary || '(no title)',
    from: e.organizer?.displayName || e.organizer?.email,
    date: when(e.start),
    text: parts.join(' '),
    url: e.htmlLink,
  };
}

export class CalendarClient implements CalendarService {
  constructor(private getToken: TokenProvider) {}

  async listEvents(opts: { window: TimeWindow; query?: string; max?: number }): Promise<Item[]> {
    const res = await googleFetch<{ items?: CalEvent[] }>(
      this.getToken,
      `${BASE}?${qs({
        timeMin: opts.window.start !== undefined ? new Date(opts.window.start).toISOString() : undefined,
        timeMax: opts.window.end !== undefined ? new Date(opts.window.end).toISOString() : undefined,
        q: opts.query,
        singleEvents: true,
        orderBy: 'startTime',
        maxResults: opts.max ?? 25,
      })}`,
    );
    return (res.items ?? []).filter((e) => e.status !== 'cancelled').map((e) => toItem(e, false));
  }

  async readEvent(id: string): Promise<Item> {
    return toItem(await googleFetch<CalEvent>(this.getToken, `${BASE}/${encodeURIComponent(id)}`), true);
  }
}
