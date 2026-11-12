import type { ActionKind } from './router';

/**
 * Write actions (Safety and approvals):
 * - A write becomes a proposed action showing exactly what will happen (recipients, body, time).
 *   The graph pauses before executing and resumes only on approval.
 * - Drafts are the default output; sending, creating or changing events needs a fresh approval.
 * - The step that executes receives only these structured fields, never raw message text.
 * - The target item is re-fetched by id immediately before acting.
 */

export interface EmailAction {
  kind: 'draft_email' | 'send_email';
  to: string[];
  cc: string[];
  subject: string;
  body: string;
  /** Gmail message id this replies to (threading + In-Reply-To). */
  replyToId?: string;
  replyToTitle?: string;
}

export interface CreateEventAction {
  kind: 'create_event';
  title: string;
  /** Epoch ms. For all-day events, local midnight of the first day. */
  start: number;
  end: number;
  allDay: boolean;
  attendees: string[];
  location?: string;
  description?: string;
}

export interface UpdateEventAction {
  kind: 'update_event';
  eventId: string;
  eventTitle: string;
  previousStart?: number;
  previousEnd?: number;
  start?: number;
  end?: number;
  title?: string;
}

export type ProposedAction = EmailAction | CreateEventAction | UpdateEventAction;

export const ACTION_LABEL: Record<ActionKind, string> = {
  draft_email: 'Save draft',
  send_email: 'Send email',
  create_event: 'Create event',
  update_event: 'Update event',
};

const EMAIL = /^[^\s@<>,;]+@[^\s@<>,;]+\.[^\s@<>,;]+$/;

export function isEmail(s: string): boolean {
  return EMAIL.test(s.trim());
}

/** "Sam Lee <sam@x.com>" → "sam@x.com"; plain addresses pass through; anything else → null. */
export function emailOf(s: string): string | null {
  const m = s.match(/<([^>]+)>/);
  const addr = (m ? (m[1] as string) : s).trim();
  return isEmail(addr) ? addr.toLowerCase() : null;
}

/** Splits a comma/semicolon list typed by the user or the model. */
export function splitList(v: unknown): string[] {
  const parts = Array.isArray(v) ? v.map(String) : typeof v === 'string' ? v.split(/[,;\n]/) : [];
  return parts.map((s) => s.trim()).filter(Boolean);
}

/** "2026-11-13" + "15:30" (local time) → epoch ms. Returns undefined on anything malformed. */
export function localDateTime(date: unknown, time?: unknown): number | undefined {
  if (typeof date !== 'string') return undefined;
  const d = date.trim().match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!d) return undefined;
  let h = 0;
  let min = 0;
  if (typeof time === 'string' && time.trim()) {
    const t = time.trim().toLowerCase().match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/);
    if (!t) return undefined;
    h = Number(t[1]);
    min = Number(t[2] ?? 0);
    if (t[3] === 'pm' && h < 12) h += 12;
    if (t[3] === 'am' && h === 12) h = 0;
    if (h > 23 || min > 59) return undefined;
  }
  const ms = new Date(Number(d[1]), Number(d[2]) - 1, Number(d[3]), h, min).getTime();
  return Number.isFinite(ms) ? ms : undefined;
}

const DAY = 24 * 60 * 60 * 1000;

/** JSON schemas for the model's structured output, per action. */
export const ACTION_SCHEMAS: Record<ActionKind, Record<string, unknown>> = {
  draft_email: {
    type: 'object',
    properties: {
      to: { type: 'array', items: { type: 'string' } },
      cc: { type: 'array', items: { type: 'string' } },
      subject: { type: 'string' },
      body: { type: 'string' },
      reply_to_id: { type: 'string' },
    },
    required: ['to', 'subject', 'body'],
  },
  send_email: {} as Record<string, unknown>, // same as draft_email, set below
  create_event: {
    type: 'object',
    properties: {
      title: { type: 'string' },
      date: { type: 'string' },
      start_time: { type: 'string' },
      end_time: { type: 'string' },
      all_day: { type: 'boolean' },
      attendees: { type: 'array', items: { type: 'string' } },
      location: { type: 'string' },
      description: { type: 'string' },
    },
    required: ['title', 'date'],
  },
  update_event: {
    type: 'object',
    properties: {
      event_id: { type: 'string' },
      date: { type: 'string' },
      start_time: { type: 'string' },
      end_time: { type: 'string' },
      title: { type: 'string' },
    },
    required: ['event_id'],
  },
};
ACTION_SCHEMAS.send_email = ACTION_SCHEMAS.draft_email;

export const ACTION_INSTRUCTIONS: Record<ActionKind, string> = {
  draft_email:
    'Fill in an email for the user to review. "to" and "cc" are email addresses, or names if no address is known. ' +
    'When replying to an email from the conversation, set reply_to_id to its id and leave "to" empty to reply to its ' +
    'sender. Write the body in the user\'s voice, plain text, short and polite, with no placeholder names.',
  send_email: '',
  create_event:
    'Fill in a calendar event. date is YYYY-MM-DD; start_time and end_time are HH:MM in 24-hour time (leave both ' +
    'empty and set all_day for all-day events). attendees are email addresses or names. Keep the title short.',
  update_event:
    'Pick the event to change from the events listed (event_id must be one of their ids). Give only what changes: ' +
    'date as YYYY-MM-DD, start_time/end_time as HH:MM in 24-hour time, or a new title.',
};
ACTION_INSTRUCTIONS.send_email = ACTION_INSTRUCTIONS.draft_email;

export interface ReplyTarget {
  id: string;
  title: string;
  /** "Name <address>" of the original sender. */
  from?: string;
}

export interface EventCandidate {
  id: string;
  title: string;
  start: number;
  end: number;
}

/** What the model returned, checked and normalized; or what is missing, as a question for the user. */
export type BuildResult = { ok: true; action: ProposedAction } | { ok: false; question: string };

export function buildAction(
  kind: ActionKind,
  raw: Record<string, unknown>,
  ctx: { replyTargets: ReplyTarget[]; events: EventCandidate[]; resolve: (name: string) => string | null },
): BuildResult {
  if (kind === 'draft_email' || kind === 'send_email') {
    const replyId = typeof raw.reply_to_id === 'string' ? raw.reply_to_id.trim() : '';
    const target = ctx.replyTargets.find((t) => t.id === replyId);
    const unresolved: string[] = [];
    const addresses = (list: string[]) =>
      list.flatMap((r) => {
        const a = emailOf(r) ?? ctx.resolve(r);
        if (!a) unresolved.push(r);
        return a ? [a] : [];
      });
    let to = addresses(splitList(raw.to));
    const cc = addresses(splitList(raw.cc));
    if (to.length === 0 && target?.from) {
      const a = emailOf(target.from);
      if (a) to = [a];
    }
    if (unresolved.length) {
      return { ok: false, question: `What is the email address for ${unresolved.join(', ')}?` };
    }
    if (to.length === 0) return { ok: false, question: 'Who should the email go to?' };
    const body = typeof raw.body === 'string' ? raw.body.trim() : '';
    if (!body) return { ok: false, question: 'What should the email say?' };
    let subject = typeof raw.subject === 'string' ? raw.subject.trim() : '';
    if (target && !subject) subject = target.title;
    if (target && subject && !/^re:/i.test(subject)) subject = `Re: ${subject}`;
    return {
      ok: true,
      action: {
        kind,
        to: [...new Set(to)],
        cc: [...new Set(cc)].filter((a) => !to.includes(a)),
        subject: subject || '(no subject)',
        body,
        ...(target ? { replyToId: target.id, replyToTitle: target.title } : {}),
      },
    };
  }

  if (kind === 'create_event') {
    const title = typeof raw.title === 'string' ? raw.title.trim() : '';
    if (!title) return { ok: false, question: 'What should the event be called?' };
    const allDay = raw.all_day === true || (!raw.start_time && !raw.end_time);
    const start = localDateTime(raw.date, allDay ? undefined : raw.start_time);
    if (start === undefined) return { ok: false, question: 'Which day and time should the event be?' };
    let end = allDay ? start + DAY : localDateTime(raw.date, raw.end_time);
    if (end === undefined || end <= start) end = start + 30 * 60 * 1000; // default 30 minutes
    const unresolved: string[] = [];
    const attendees = splitList(raw.attendees).flatMap((r) => {
      const a = emailOf(r) ?? ctx.resolve(r);
      if (!a) unresolved.push(r);
      return a ? [a] : [];
    });
    if (unresolved.length) return { ok: false, question: `What is the email address for ${unresolved.join(', ')}?` };
    return {
      ok: true,
      action: {
        kind,
        title,
        start,
        end,
        allDay,
        attendees: [...new Set(attendees)],
        ...(typeof raw.location === 'string' && raw.location.trim() ? { location: raw.location.trim() } : {}),
        ...(typeof raw.description === 'string' && raw.description.trim() ? { description: raw.description.trim() } : {}),
      },
    };
  }

  // update_event
  const id = typeof raw.event_id === 'string' ? raw.event_id.trim() : '';
  const event = ctx.events.find((e) => e.id === id);
  if (!event) {
    return {
      ok: false,
      question: ctx.events.length
        ? 'Which event should I change? Tell me its name or time.'
        : "I couldn't find that event in the next two weeks. Which day is it on?",
    };
  }
  const duration = event.end - event.start;
  const day = typeof raw.date === 'string' && raw.date.trim() ? raw.date : localIsoDate(event.start);
  const startTime = typeof raw.start_time === 'string' && raw.start_time.trim() ? raw.start_time : undefined;
  const dateChanged = typeof raw.date === 'string' && raw.date.trim() !== '' && raw.date.trim() !== localIsoDate(event.start);
  let start: number | undefined;
  let end: number | undefined;
  if (startTime || dateChanged) {
    start = localDateTime(day, startTime ?? localTime(event.start));
    if (start === undefined) return { ok: false, question: 'What new day or time should the event move to?' };
    const endTime = typeof raw.end_time === 'string' && raw.end_time.trim() ? raw.end_time : undefined;
    end = endTime ? localDateTime(day, endTime) : undefined;
    if (end === undefined || end <= start) end = start + duration; // keep the length
  }
  const title = typeof raw.title === 'string' && raw.title.trim() && raw.title.trim() !== event.title ? raw.title.trim() : undefined;
  if (start === undefined && !title) return { ok: false, question: `What should change about "${event.title}"?` };
  return {
    ok: true,
    action: {
      kind: 'update_event',
      eventId: event.id,
      eventTitle: event.title,
      previousStart: event.start,
      previousEnd: event.end,
      ...(start !== undefined ? { start, end } : {}),
      ...(title ? { title } : {}),
    },
  };
}

function pad(n: number): string {
  return String(n).padStart(2, '0');
}

export function localIsoDate(t: number): string {
  const d = new Date(t);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function localTime(t: number): string {
  const d = new Date(t);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function when(start: number, end: number, allDay = false): string {
  const day = new Date(start).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  if (allDay) return `${day}, all day`;
  return `${day}, ${localTime(start)}–${localTime(end)}`;
}

/** The approval text: exactly what will happen, in Markdown. */
export function describeAction(a: ProposedAction): string {
  switch (a.kind) {
    case 'draft_email':
    case 'send_email': {
      const verb = a.kind === 'send_email' ? '**Send** this email' : 'Save this **draft** in Gmail (nothing is sent)';
      const lines = [
        `${verb}:`,
        '',
        `- **To:** ${a.to.join(', ')}`,
        ...(a.cc.length ? [`- **Cc:** ${a.cc.join(', ')}`] : []),
        `- **Subject:** ${a.subject}`,
        ...(a.replyToTitle ? [`- **Replying to:** ${a.replyToTitle}`] : []),
        '',
        ...a.body.split('\n').map((l) => `> ${l}`),
      ];
      return lines.join('\n');
    }
    case 'create_event':
      return [
        'Create this event in your Google Calendar:',
        '',
        `- **${a.title}**`,
        `- **When:** ${when(a.start, a.end, a.allDay)}`,
        ...(a.location ? [`- **Where:** ${a.location}`] : []),
        ...(a.attendees.length ? [`- **Invites:** ${a.attendees.join(', ')} (Google emails them an invitation)`] : []),
      ].join('\n');
    case 'update_event':
      return [
        `Change **${a.eventTitle}**:`,
        '',
        ...(a.start !== undefined && a.end !== undefined && a.previousStart !== undefined && a.previousEnd !== undefined
          ? [`- **From:** ${when(a.previousStart, a.previousEnd)}`, `- **To:** ${when(a.start, a.end)}`]
          : []),
        ...(a.title ? [`- **New title:** ${a.title}`] : []),
        '- Guests get Google\'s update email.',
      ].join('\n');
  }
}

/** Builds an RFC 2822 message for the Gmail API (UTF-8, base64 body), returned as base64url. */
export function buildRawEmail(
  a: EmailAction,
  reply: { messageId?: string; references?: string } | undefined,
  encode: { utf8: (s: string) => Uint8Array; base64: (b: Uint8Array) => string },
): string {
  const b64 = (s: string) => encode.base64(encode.utf8(s));
  // Non-ASCII subjects use RFC 2047 encoded words.
  const subject = /^[\x20-\x7e]*$/.test(a.subject) ? a.subject : `=?UTF-8?B?${b64(a.subject)}?=`;
  const headers = [
    `To: ${a.to.join(', ')}`,
    ...(a.cc.length ? [`Cc: ${a.cc.join(', ')}`] : []),
    `Subject: ${subject}`,
    'MIME-Version: 1.0',
    'Content-Type: text/plain; charset="UTF-8"',
    'Content-Transfer-Encoding: base64',
    ...(reply?.messageId ? [`In-Reply-To: ${reply.messageId}`, `References: ${[reply.references, reply.messageId].filter(Boolean).join(' ')}`] : []),
  ];
  const body = (b64(a.body).match(/.{1,76}/g) ?? []).join('\r\n');
  const message = `${headers.join('\r\n')}\r\n\r\n${body}`;
  return b64(message).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** The action as editable text fields, for the approval card's Edit mode. */
export function editableFields(a: ProposedAction): Record<string, string> {
  switch (a.kind) {
    case 'draft_email':
    case 'send_email':
      return { to: a.to.join(', '), cc: a.cc.join(', '), subject: a.subject, body: a.body };
    case 'create_event':
      return {
        title: a.title,
        date: localIsoDate(a.start),
        start: a.allDay ? '' : localTime(a.start),
        end: a.allDay ? '' : localTime(a.end),
        attendees: a.attendees.join(', '),
        location: a.location ?? '',
      };
    case 'update_event':
      return {
        title: a.title ?? a.eventTitle,
        date: localIsoDate(a.start ?? a.previousStart ?? Date.now()),
        start: localTime(a.start ?? a.previousStart ?? Date.now()),
        end: localTime(a.end ?? a.previousEnd ?? Date.now()),
      };
  }
}

/** Applies the user's edits, with the same checks as the model's output. */
export function applyEdits(a: ProposedAction, f: Record<string, string>): { ok: true; action: ProposedAction } | { ok: false; error: string } {
  if (a.kind === 'draft_email' || a.kind === 'send_email') {
    const bad = [...splitList(f.to), ...splitList(f.cc)].filter((x) => !emailOf(x));
    if (bad.length) return { ok: false, error: `Not an email address: ${bad.join(', ')}` };
    const to = splitList(f.to).map((x) => emailOf(x) as string);
    if (!to.length) return { ok: false, error: 'Add at least one recipient.' };
    if (!(f.body ?? '').trim()) return { ok: false, error: 'The email is empty.' };
    return {
      ok: true,
      action: { ...a, to, cc: splitList(f.cc).map((x) => emailOf(x) as string), subject: (f.subject ?? '').trim() || '(no subject)', body: (f.body ?? '').trim() },
    };
  }
  if (a.kind === 'create_event') {
    const allDay = !(f.start ?? '').trim();
    const start = localDateTime(f.date, allDay ? undefined : f.start);
    if (start === undefined) return { ok: false, error: 'Use a date like 2026-11-20 and a time like 15:30.' };
    let end = allDay ? start + DAY : localDateTime(f.date, f.end);
    if (end === undefined || end <= start) end = start + 30 * 60 * 1000;
    const bad = splitList(f.attendees).filter((x) => !emailOf(x));
    if (bad.length) return { ok: false, error: `Not an email address: ${bad.join(', ')}` };
    if (!(f.title ?? '').trim()) return { ok: false, error: 'Give the event a title.' };
    return {
      ok: true,
      action: {
        ...a,
        title: (f.title ?? '').trim(),
        start,
        end,
        allDay,
        attendees: splitList(f.attendees).map((x) => emailOf(x) as string),
        ...(f.location?.trim() ? { location: f.location.trim() } : { location: undefined }),
      },
    };
  }
  const u = a as UpdateEventAction;
  const start = localDateTime(f.date, f.start);
  const end = localDateTime(f.date, f.end);
  if (start === undefined || end === undefined || end <= start) {
    return { ok: false, error: 'Use a date like 2026-11-20 and times like 15:00 and 15:30.' };
  }
  const title = (f.title ?? '').trim();
  const updated: UpdateEventAction = { ...u, start, end, title: title && title !== u.eventTitle ? title : undefined };
  return { ok: true, action: updated };
}
