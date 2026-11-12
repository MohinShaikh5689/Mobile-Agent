import { extractKeywords, routeRequest } from '../src/core/router';
import { parseTimeWindow } from '../src/core/timeWindow';

// Wednesday 11 Nov 2026, 10:00 local time.
const NOW = new Date(2026, 10, 11, 10, 0, 0).getTime();
const day = (m: number, d: number) => new Date(2026, m, d).getTime();

describe('parseTimeWindow', () => {
  it.each([
    ['what did Sam send me today', day(10, 11), day(10, 12)],
    ['mail from yesterday', day(10, 10), day(10, 11)],
    ["what's on tomorrow", day(10, 12), day(10, 13)],
    ['mail from last tuesday', day(10, 10), day(10, 11)],
    ['meeting on monday', day(10, 9), day(10, 10)],
    ['next friday', day(10, 20), day(10, 21)],
    ['this week', day(10, 9), day(10, 16)],
    ['last week', day(10, 2), day(10, 9)],
    ['in march', day(2, 1), day(3, 1)],
  ])('%s', (text, start, end) => {
    expect(parseTimeWindow(text, NOW)?.window).toEqual({ start, end });
  });

  it('puts a future month name in the past year', () => {
    expect(parseTimeWindow('the invoice in december', NOW)?.window.start).toBe(new Date(2025, 11, 1).getTime());
  });

  it('returns null with no date', () => {
    expect(parseTimeWindow('find the contract with the 30-day notice', NOW)).toBeNull();
  });
});

describe('routeRequest', () => {
  // The sample questions from the project doc, with the route each should take.
  it.each([
    ["What's on my calendar tomorrow?", 'structured', ['calendar']],
    ['Any unread mail from Sam?', 'structured', ['mail']],
    ['what did Sam send me today', 'structured', ['mail']],
    ['find the contract with the 30-day notice', 'fuzzy', ['docs']],
    ['the thread about the visa paperwork', 'fuzzy', ['mail']],
    ['what the vendor said about pricing', 'fuzzy', ['mail', 'docs']],
    ['Draft a reply to Sam', 'action', ['mail']],
    ['Reschedule my 3pm meeting and draft a note to the attendees', 'multi_step', ['mail', 'calendar']],
    ['Did Sam reply to my email?', 'structured', ['mail']],
  ])('%s → %s', (text, route, services) => {
    const r = routeRequest(text, NOW);
    expect(r.route).toBe(route);
    expect([...r.services].sort()).toEqual([...services].sort());
  });

  it('does not treat a question about sending as a send request', () => {
    expect(routeRequest('what did the landlord send last week', NOW).route).not.toBe('action');
  });

  it('carries the time window', () => {
    const r = routeRequest('what did the vendor say about pricing last week', NOW);
    expect(r.route).toBe('fuzzy');
    expect(r.window).toEqual({ start: day(10, 2), end: day(10, 9) });
    expect(r.keywords).toEqual(['vendor', 'pricing']);
  });
});

describe('extractKeywords', () => {
  it('drops stop words, duplicates and the date phrase', () => {
    expect(extractKeywords('find the email about the visa visa paperwork from yesterday', 'yesterday')).toEqual([
      'visa',
      'paperwork',
    ]);
  });
});

describe('general conversation', () => {
  it.each(['hi', 'Hello!', 'thanks', 'how are you?', 'what can you do?', 'good morning'])('%s → general', (q) => {
    const r = routeRequest(q, NOW);
    expect(r.route).toBe('general');
    expect(r.ambiguous).toBeFalsy();
  });

  it.each(['explain how a VPN works', 'list some good books about habits', 'where does pasta come from', 'what is the capital of France'])(
    '%s → unsure, for the classifier',
    (q) => {
      expect(routeRequest(q, NOW).ambiguous).toBe(true);
    },
  );

  it.each([
    "What's on my calendar tomorrow?",
    'Any unread mail from Sam?',
    'did the landlord send the lease',
    'find the contract with the 30-day notice',
    'what did Priya say about my visa',
  ])('%s → not sent to the classifier', (q) => {
    expect(routeRequest(q, NOW).ambiguous).toBeFalsy();
  });
});
