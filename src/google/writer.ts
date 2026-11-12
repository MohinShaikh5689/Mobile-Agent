import type { CreateEventAction, EmailAction, UpdateEventAction } from '../core/actions';
import { buildRawEmail } from '../core/actions';
import { bytesToBase64, utf8Encode } from '../core/encoding';
import type { WriteResult, WriteService } from '../core/services';
import type { TokenProvider } from './http';
import { googleFetch, qs } from './http';

const GMAIL = 'https://gmail.googleapis.com/gmail/v1/users/me';
const CAL = 'https://www.googleapis.com/calendar/v3/calendars/primary/events';

type Reply = { threadId: string; messageId?: string; references?: string };

function raw(a: EmailAction, reply?: Reply): string {
  return buildRawEmail(a, reply, { utf8: utf8Encode, base64: bytesToBase64 });
}

function eventTime(t: number, allDay: boolean): { date?: string; dateTime?: string } {
  if (!allDay) return { dateTime: new Date(t).toISOString() };
  const d = new Date(t);
  const p = (n: number) => String(n).padStart(2, '0');
  return { date: `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}` };
}

/** Gmail drafts/sending and Calendar events. Called only after the user approved the action. */
export class GoogleWriter implements WriteService {
  constructor(private getToken: TokenProvider) {}

  async replyHeaders(messageId: string) {
    const m = await googleFetch<{ threadId: string; payload?: { headers?: { name: string; value: string }[] } }>(
      this.getToken,
      `${GMAIL}/messages/${messageId}?${qs({ format: 'metadata', metadataHeaders: ['Message-ID', 'References', 'Subject'] })}`,
    );
    const h = (n: string) => m.payload?.headers?.find((x) => x.name.toLowerCase() === n.toLowerCase())?.value;
    return { threadId: m.threadId, messageId: h('Message-ID'), references: h('References'), subject: h('Subject') ?? '' };
  }

  async createDraft(a: EmailAction, reply?: Reply): Promise<WriteResult> {
    const res = await googleFetch<{ id: string; message?: { id: string } }>(this.getToken, `${GMAIL}/drafts`, {
      method: 'POST',
      body: { message: { raw: raw(a, reply), ...(reply ? { threadId: reply.threadId } : {}) } },
    });
    return { id: res.id, url: 'https://mail.google.com/mail/u/0/#drafts' };
  }

  async sendEmail(a: EmailAction, reply?: Reply): Promise<WriteResult> {
    const res = await googleFetch<{ id: string; threadId: string }>(this.getToken, `${GMAIL}/messages/send`, {
      method: 'POST',
      body: { raw: raw(a, reply), ...(reply ? { threadId: reply.threadId } : {}) },
    });
    return { id: res.id, url: `https://mail.google.com/mail/u/0/#sent/${res.threadId}` };
  }

  async createEvent(a: CreateEventAction): Promise<WriteResult> {
    const res = await googleFetch<{ id: string; htmlLink?: string }>(
      this.getToken,
      `${CAL}?${qs({ sendUpdates: a.attendees.length ? 'all' : 'none' })}`,
      {
        method: 'POST',
        body: {
          summary: a.title,
          start: eventTime(a.start, a.allDay),
          end: eventTime(a.end, a.allDay),
          ...(a.location ? { location: a.location } : {}),
          ...(a.description ? { description: a.description } : {}),
          ...(a.attendees.length ? { attendees: a.attendees.map((email) => ({ email })) } : {}),
        },
      },
    );
    return { id: res.id, url: res.htmlLink };
  }

  async updateEvent(a: UpdateEventAction): Promise<WriteResult> {
    const res = await googleFetch<{ id: string; htmlLink?: string }>(
      this.getToken,
      `${CAL}/${encodeURIComponent(a.eventId)}?${qs({ sendUpdates: 'all' })}`,
      {
        method: 'PATCH',
        body: {
          ...(a.title ? { summary: a.title } : {}),
          ...(a.start !== undefined && a.end !== undefined
            ? { start: eventTime(a.start, false), end: eventTime(a.end, false) }
            : {}),
        },
      },
    );
    return { id: res.id, url: res.htmlLink };
  }
}
