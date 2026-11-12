import type { CreateEventAction, EmailAction, ProposedAction, UpdateEventAction } from '../src/core/actions';
import { applyEdits, buildAction, buildRawEmail, describeAction, editableFields, localDateTime } from '../src/core/actions';
import { bytesToBase64, decodeBase64Url, utf8Encode } from '../src/core/encoding';
import { DurableCheckpointSaver } from '../src/core/graph/checkpointer';
import type { ChatMessage, ChatModel, ChatResult } from '../src/core/llm/types';
import { Agent } from '../src/core/orchestrator';
import { HybridRetriever } from '../src/core/retriever';
import { detectAction, routeRequest } from '../src/core/router';
import type { KeyValueStore, Services, WriteService } from '../src/core/services';
import type { Item } from '../src/core/types';
import { InMemoryVectorIndex } from '../src/core/vectorIndex';

const NOW = new Date(2026, 10, 12, 10, 0).getTime(); // Thu 12 Nov 2026, 10:00
const at = (d: number, h: number, m = 0) => new Date(2026, 10, d, h, m).getTime();

class MemoryKV implements KeyValueStore {
  data = new Map<string, string>();
  async get(k: string) {
    return this.data.get(k) ?? null;
  }
  async set(k: string, v: string) {
    this.data.set(k, v);
  }
  async delete(k: string) {
    this.data.delete(k);
  }
}

const noResolve = { replyTargets: [], events: [], resolve: () => null };

describe('buildAction', () => {
  it('builds an email and resolves names', () => {
    const r = buildAction('draft_email', { to: ['Priya', 'sam@x.test'], subject: 'Lunch', body: 'Lunch tomorrow?' }, {
      ...noResolve,
      resolve: (n) => (n === 'Priya' ? 'priya@x.test' : null),
    });
    expect(r).toEqual({
      ok: true,
      action: { kind: 'draft_email', to: ['priya@x.test', 'sam@x.test'], cc: [], subject: 'Lunch', body: 'Lunch tomorrow?' },
    });
  });

  it('asks instead of guessing when a recipient or the body is missing', () => {
    expect(buildAction('draft_email', { to: ['Priya'], subject: 's', body: 'b' }, noResolve)).toEqual({
      ok: false,
      question: 'What is the email address for Priya?',
    });
    expect(buildAction('draft_email', { to: [], subject: 's', body: 'b' }, noResolve)).toMatchObject({ ok: false, question: 'Who should the email go to?' });
    expect(buildAction('send_email', { to: ['a@x.test'], subject: 's', body: '' }, noResolve)).toMatchObject({ ok: false });
  });

  it('replies to the sender of the chosen email, with Re: subject', () => {
    const r = buildAction(
      'draft_email',
      { to: [], subject: 'Visa forms', body: 'Thanks!', reply_to_id: 'm1' },
      { ...noResolve, replyTargets: [{ id: 'm1', title: 'Visa forms', from: 'Priya Rao <Priya@x.test>' }] },
    );
    expect(r).toMatchObject({ ok: true, action: { to: ['priya@x.test'], subject: 'Re: Visa forms', replyToId: 'm1' } });
  });

  it('builds events: times, all-day, default length', () => {
    expect(buildAction('create_event', { title: 'Dentist', date: '2026-11-13', start_time: '3pm' }, noResolve)).toMatchObject({
      ok: true,
      action: { kind: 'create_event', start: at(13, 15), end: at(13, 15, 30), allDay: false },
    });
    expect(buildAction('create_event', { title: 'Trip', date: '2026-11-20', all_day: true }, noResolve)).toMatchObject({
      ok: true,
      action: { start: at(20, 0), end: at(21, 0), allDay: true },
    });
    expect(buildAction('create_event', { title: 'X', date: 'tomorrow' }, noResolve)).toMatchObject({ ok: false });
  });

  it('moves an event and keeps its length; refuses unknown ids', () => {
    const events = [{ id: 'e1', title: 'Design review', start: at(12, 15), end: at(12, 16) }];
    expect(buildAction('update_event', { event_id: 'e1', start_time: '16:30' }, { ...noResolve, events })).toMatchObject({
      ok: true,
      action: { kind: 'update_event', eventId: 'e1', start: at(12, 16, 30), end: at(12, 17, 30), previousStart: at(12, 15) },
    });
    expect(buildAction('update_event', { event_id: 'nope', start_time: '16:30' }, { ...noResolve, events })).toMatchObject({ ok: false });
  });

  it('parses 12-hour and 24-hour times', () => {
    expect(localDateTime('2026-11-12', '12am')).toBe(at(12, 0));
    expect(localDateTime('2026-11-12', '12:15 pm')).toBe(at(12, 12, 15));
    expect(localDateTime('2026-11-12', '25:00')).toBeUndefined();
  });
});

describe('approval text and edits', () => {
  const email: EmailAction = { kind: 'send_email', to: ['a@x.test'], cc: [], subject: 'Hi', body: 'Line one\nLine two' };

  it('says exactly what will happen', () => {
    expect(describeAction(email)).toContain('**Send** this email');
    expect(describeAction({ ...email, kind: 'draft_email' })).toContain('nothing is sent');
    expect(describeAction(email)).toContain('> Line two');
    const ev: CreateEventAction = { kind: 'create_event', title: 'Sync', start: at(13, 15), end: at(13, 15, 30), allDay: false, attendees: ['b@x.test'] };
    expect(describeAction(ev)).toMatch(/Fri, Nov 13, 15:00–15:30/);
    expect(describeAction(ev)).toContain('Google emails them an invitation');
  });

  it('round-trips through the edit form and validates edits', () => {
    const f = editableFields(email);
    expect(applyEdits(email, { ...f, subject: 'Hello' })).toMatchObject({ ok: true, action: { subject: 'Hello' } });
    expect(applyEdits(email, { ...f, to: 'not an address' })).toMatchObject({ ok: false });
    const up: UpdateEventAction = { kind: 'update_event', eventId: 'e1', eventTitle: 'Review', previousStart: at(12, 15), previousEnd: at(12, 16), start: at(12, 16), end: at(12, 17) };
    expect(applyEdits(up, { ...editableFields(up), start: '09:00', end: '09:45' })).toMatchObject({ ok: true, action: { start: at(12, 9), end: at(12, 9, 45) } });
  });
});

describe('Gmail message', () => {
  it('builds a UTF-8 reply with threading headers', () => {
    const raw = buildRawEmail(
      { kind: 'draft_email', to: ['a@x.test'], cc: ['c@x.test'], subject: 'Re: Café ☕', body: 'Grazie — à bientôt' },
      { messageId: '<abc@mail>', references: '<root@mail>' },
      { utf8: utf8Encode, base64: bytesToBase64 },
    );
    expect(raw).not.toMatch(/[+/=]/); // base64url
    const message = decodeBase64Url(raw);
    expect(message).toContain('To: a@x.test\r\nCc: c@x.test\r\nSubject: =?UTF-8?B?');
    expect(message).toContain('In-Reply-To: <abc@mail>\r\nReferences: <root@mail> <abc@mail>');
    const body = message.split('\r\n\r\n')[1] as string;
    expect(decodeBase64Url(body.replace(/\r\n/g, ''))).toBe('Grazie — à bientôt');
  });
});

describe('routing for actions and follow-ups', () => {
  it.each([
    ['Draft a reply to Sam', 'draft_email'],
    ['Send an email to Priya saying I am late', 'send_email'],
    ['Schedule a call with Priya tomorrow at 3pm', 'create_event'],
    ['Reschedule my 3pm meeting to 4pm', 'update_event'],
  ])('%s → %s', (q, kind) => {
    expect(detectAction(q.toLowerCase())).toBe(kind);
    expect(routeRequest(q, NOW).action).toBe(kind);
  });

  it('keeps the previous topic for follow-ups that name no service', () => {
    const r = routeRequest('See in next week buddy', NOW, { carry: ['calendar'] });
    expect(r).toMatchObject({ route: 'structured', services: ['calendar'] });
    expect(r.reason).toMatch(/continues the previous topic/);
    expect(routeRequest('thanks!', NOW, { carry: ['calendar'] }).route).toBe('general');
    expect(routeRequest('explain how compound interest works in simple words please', NOW, { carry: ['calendar'] }).ambiguous).toBe(true);
  });

  it('lists recent Drive files instead of searching by meaning', () => {
    expect(routeRequest('Anything new in drive??', NOW)).toMatchObject({ route: 'structured', services: ['docs'], listing: true });
  });
});

// ---- End-to-end through the graph, with a fake Google ----

function scripted(chat: ChatResult[], json: unknown[] = []) {
  const jsonSeen: ChatMessage[][] = [];
  const model: ChatModel = {
    chat: async () => chat.shift() ?? { content: 'ok', toolCalls: [] },
    async chatJson(messages) {
      jsonSeen.push(messages);
      if (!json.length) throw new Error('no scripted JSON');
      return json.shift();
    },
  };
  return { model, jsonSeen };
}

function fakeGoogle() {
  const calls: { op: string; arg: unknown }[] = [];
  const priyaMail: Item = { source: 'mail', id: 'm1', title: 'Visa forms', text: 'Forms attached, can you confirm?', date: NOW - 3600_000, from: 'Priya Rao <priya@x.test>' };
  const review: Item = { source: 'calendar', id: 'e1', title: 'Design review', text: '15:00–16:00', date: at(12, 15) };
  const write: WriteService = {
    replyHeaders: async (id) => {
      calls.push({ op: 'replyHeaders', arg: id });
      return { threadId: 't1', messageId: '<m1@mail>', subject: 'Visa forms' };
    },
    createDraft: async (a, reply) => {
      calls.push({ op: 'createDraft', arg: { a, reply } });
      return { id: 'd1', url: 'https://mail.google.com/mail/u/0/#drafts' };
    },
    sendEmail: async (a) => {
      calls.push({ op: 'sendEmail', arg: a });
      return { id: 's1' };
    },
    createEvent: async (a) => {
      calls.push({ op: 'createEvent', arg: a });
      return { id: 'new1', url: 'https://calendar.google.com/x' };
    },
    updateEvent: async (a) => {
      calls.push({ op: 'updateEvent', arg: a });
      return { id: a.eventId };
    },
  };
  const services: Services = {
    mail: {
      search: async (q) => {
        calls.push({ op: 'mail.search', arg: q });
        return q.from === 'Priya' ? [priyaMail] : [priyaMail];
      },
      read: async (id) => {
        calls.push({ op: 'mail.read', arg: id });
        return priyaMail;
      },
    },
    docs: {
      search: async (q) => {
        calls.push({ op: 'docs.search', arg: q });
        return [{ source: 'docs', id: 'f1', title: 'Budget.xlsx', text: '', date: NOW - 86_400_000, from: 'Me' }];
      },
      read: async () => priyaMail,
    },
    calendar: {
      listEvents: async (q) => {
        calls.push({ op: 'calendar.list', arg: q });
        return [review];
      },
      readEvent: async (id) => {
        calls.push({ op: 'calendar.read', arg: id });
        return review;
      },
    },
    write,
  };
  return { services, calls };
}

function makeAgent(model: ChatModel, services: Services, kv = new MemoryKV()) {
  return {
    kv,
    agent: new Agent({
      model,
      retriever: new HybridRetriever({
        index: new InMemoryVectorIndex(),
        embedder: { embed: async (t) => t.map(() => [1]) },
        live: { mail: (q) => services.mail.search(q), docs: async () => [] },
        coverage: async () => ({}),
      }),
      services,
      loadCoverage: async () => ({}),
      kv,
      checkpointer: new DurableCheckpointSaver(kv),
      clock: { now: () => NOW },
    }),
  };
}

describe('write actions through the graph', () => {
  it('proposes, waits, and writes only after approval', async () => {
    const { services, calls } = fakeGoogle();
    const { model } = scripted([], [{ to: ['sam@x.test'], subject: 'Lunch', body: 'Lunch tomorrow at 1?' }]);
    const { agent } = makeAgent(model, services);

    const r = await agent.ask('Draft an email to sam@x.test about lunch tomorrow');
    expect(r.path).toEqual(['contextualize', 'router', 'propose']);
    expect(r.pending?.action).toMatchObject({ kind: 'draft_email', to: ['sam@x.test'], subject: 'Lunch' });
    expect(r.answer).toContain('nothing is sent');
    expect(calls.filter((c) => c.op === 'createDraft')).toHaveLength(0); // nothing written yet
    expect((await agent.pending())?.threadId).toBe(r.threadId);

    const done = await agent.approve(r.threadId, r.pending?.action);
    expect(done.path).toEqual(['execute']);
    expect(done.answer).toMatch(/^Draft saved in Gmail\. Nothing was sent\. \[Open\]/);
    expect(calls.filter((c) => c.op === 'createDraft')).toHaveLength(1);
    expect(await agent.pending()).toBeUndefined();
  });

  it('writes what the user edited, not what was proposed', async () => {
    const { services, calls } = fakeGoogle();
    const { model } = scripted([], [{ to: ['sam@x.test'], subject: 'Lunch', body: 'Lunch?' }]);
    const { agent } = makeAgent(model, services);
    const r = await agent.ask('Draft an email to sam@x.test about lunch');
    const edited = applyEdits(r.pending?.action as ProposedAction, { to: 'sam@x.test', cc: '', subject: 'Lunch on Friday', body: 'Friday instead?' });
    if (!edited.ok) throw new Error(edited.error);
    await agent.approve(r.threadId, edited.action);
    const draft = calls.find((c) => c.op === 'createDraft')?.arg as { a: EmailAction };
    expect(draft.a).toMatchObject({ subject: 'Lunch on Friday', body: 'Friday instead?' });
  });

  it('cancel writes nothing and clears the proposal', async () => {
    const { services, calls } = fakeGoogle();
    const { model } = scripted([], [{ to: ['sam@x.test'], subject: 'x', body: 'y' }]);
    const { agent } = makeAgent(model, services);
    const r = await agent.ask('Send an email to sam@x.test saying hi');
    expect(r.pending?.action.kind).toBe('send_email');
    expect(await agent.cancel(r.threadId)).toMatch(/Nothing was changed/);
    expect(calls.some((c) => c.op === 'sendEmail')).toBe(false);
    expect(await agent.pending()).toBeUndefined();
    await expect(agent.approve(r.threadId)).rejects.toThrow(/no longer waiting/);
  });

  it('a proposal survives an app restart and can be approved after it', async () => {
    const { services, calls } = fakeGoogle();
    const kv = new MemoryKV();
    const first = makeAgent(scripted([], [{ to: ['sam@x.test'], subject: 'x', body: 'y' }]).model, services, kv).agent;
    const r = await first.ask('Draft an email to sam@x.test saying hi');
    const second = makeAgent(scripted([]).model, services, kv).agent; // "restart"
    const pending = await second.pending();
    expect(pending?.action.kind).toBe('draft_email');
    await second.approve(pending?.threadId as string);
    expect(calls.filter((c) => c.op === 'createDraft')).toHaveLength(1);
  });

  it('replies to an email from the conversation: reads it, re-fetches it before writing', async () => {
    const { services, calls } = fakeGoogle();
    const { model, jsonSeen } = scripted(
      [{ content: 'Priya sent the visa forms [1].', toolCalls: [] }],
      [
        { relevant: true },
        { standalone: 'Reply to the Visa forms email from Priya saying thanks, confirmed' },
        { to: [], subject: 'Visa forms', body: 'Thanks, confirmed!', reply_to_id: 'm1' },
      ],
    );
    const { agent } = makeAgent(model, services);
    // A recall question puts m1 into the conversation's items mentioned.
    await agent.ask('find the email about the visa forms');
    const r = await agent.ask('reply to that email saying thanks, confirmed');
    expect(r.route.action).toBe('draft_email');
    expect(calls.some((c) => c.op === 'mail.read' && c.arg === 'm1')).toBe(true);
    expect(jsonSeen.at(-1)?.[1]?.content).toContain('email being replied to');
    expect(r.pending?.action).toMatchObject({ to: ['priya@x.test'], subject: 'Re: Visa forms', replyToId: 'm1' });
    await agent.approve(r.threadId);
    const draft = calls.find((c) => c.op === 'createDraft')?.arg as { reply: { threadId: string } };
    expect(calls.some((c) => c.op === 'replyHeaders')).toBe(true);
    expect(draft.reply.threadId).toBe('t1');
  });

  it('creates an event, finding attendee addresses by name', async () => {
    const { services, calls } = fakeGoogle();
    const { model } = scripted([], [{ title: 'Call with Priya', date: '2026-11-13', start_time: '15:00', end_time: '15:30', attendees: ['Priya'] }]);
    const { agent } = makeAgent(model, services);
    const r = await agent.ask('Schedule a call with Priya tomorrow at 3pm');
    expect(calls.some((c) => c.op === 'mail.search' && (c.arg as { from?: string }).from === 'Priya')).toBe(true);
    expect(r.pending?.action).toMatchObject({ kind: 'create_event', attendees: ['priya@x.test'], start: at(13, 15) });
    const done = await agent.approve(r.threadId);
    expect(done.answer).toMatch(/Added \*\*Call with Priya\*\* to your calendar\. Invitations were sent\./);
  });

  it('reschedules: offers only real events, re-checks the event before changing it', async () => {
    const { services, calls } = fakeGoogle();
    const { model, jsonSeen } = scripted([], [{ event_id: 'e1', start_time: '16:00' }]);
    const { agent } = makeAgent(model, services);
    const r = await agent.ask('Reschedule my 3pm meeting to 4pm');
    expect(jsonSeen[0]?.[1]?.content).toContain('id=e1');
    expect(r.pending?.action).toMatchObject({ kind: 'update_event', eventId: 'e1', start: at(12, 16), end: at(12, 17) });
    await agent.approve(r.threadId);
    const ops = calls.map((c) => c.op);
    expect(ops.indexOf('calendar.read')).toBeLessThan(ops.indexOf('updateEvent'));
  });

  it('asks a question instead of proposing when something is missing', async () => {
    const { services } = fakeGoogle();
    const { model } = scripted([], [{ to: [], subject: 'hi', body: 'hello' }]);
    const { agent } = makeAgent(model, services);
    const r = await agent.ask('Draft an email saying hello');
    expect(r.pending).toBeUndefined();
    expect(r.answer).toBe('Who should the email go to?');
  });

  it('tells the user what is not supported', async () => {
    const { services } = fakeGoogle();
    const { agent } = makeAgent(scripted([]).model, services);
    const r = await agent.ask('Cancel my 3pm meeting');
    expect(r.answer).toMatch(/not supported yet/);
  });
});

describe('fixes from real use', () => {
  it('a follow-up after a calendar question looks at the calendar, not general chat', async () => {
    const { services, calls } = fakeGoogle();
    const { model } = scripted([
      { content: 'Design review today at 15:00.', toolCalls: [] },
      { content: 'Design review next week.', toolCalls: [] },
    ]);
    const { agent } = makeAgent(model, services);
    await agent.ask('Any upcoming events in my calendar??');
    const r = await agent.ask('See in next week buddy');
    expect(r.path).toEqual(['contextualize', 'router', 'specialist']);
    expect(calls.filter((c) => c.op === 'calendar.list')).toHaveLength(2);
  });

  it('"added recently" on the calendar looks ahead, not at last week', async () => {
    const { services, calls } = fakeGoogle();
    const { agent } = makeAgent(scripted([{ content: 'ok', toolCalls: [] }]).model, services);
    await agent.ask('See again in my calendar, some event is added recently');
    const window = (calls.find((c) => c.op === 'calendar.list')?.arg as { window: { start: number } }).window;
    expect(window.start).toBe(NOW);
  });

  it('"anything new in Drive" lists recently changed files', async () => {
    const { services, calls } = fakeGoogle();
    const { agent } = makeAgent(scripted([{ content: '- **Budget.xlsx**', toolCalls: [] }]).model, services);
    const r = await agent.ask('Anything new in drive??');
    expect(r.path).toEqual(['contextualize', 'router', 'specialist']);
    expect(calls.find((c) => c.op === 'docs.search')?.arg).toMatchObject({ keywords: [] });
    expect(r.steps[0]).toMatchObject({ name: 'search_docs', ok: true });
  });
});
