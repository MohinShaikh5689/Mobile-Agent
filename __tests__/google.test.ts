import { splitText } from '../src/core/chunker';
import { decodeBase64Url, htmlToText } from '../src/core/encoding';
import { buildDriveQuery } from '../src/google/drive';
import { buildGmailQuery, messageBody } from '../src/google/gmail';

declare const Buffer: { from(s: string, enc: string): { toString(enc: string): string } };

function b64url(s: string): string {
  // Test-only encoder (Node has Buffer; the app does not).
  return Buffer.from(s, 'utf8').toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

describe('encoding', () => {
  it('decodes base64url UTF-8, including non-ASCII', () => {
    const text = 'Visa — überprüft ✓ 日本 🎉';
    expect(decodeBase64Url(b64url(text))).toBe(text);
  });

  it('turns HTML into text', () => {
    expect(htmlToText('<p>Hello&nbsp;<b>Sam</b></p><style>x{}</style><p>Bye</p>')).toBe('Hello Sam\nBye');
  });
});

describe('gmail', () => {
  it('builds a query with OR-ed keywords, epoch-second dates and filters', () => {
    expect(
      buildGmailQuery({ keywords: ['visa', 'paperwork'], window: { start: 1_700_000_000_000, end: 1_700_086_400_000 }, unreadOnly: true, from: 'Sam Lee' }),
    ).toBe('{visa paperwork} from:"Sam Lee" after:1700000000 before:1700086400 is:unread -in:spam -in:trash');
  });

  it('prefers the text/plain part', () => {
    const body = messageBody({
      id: '1',
      threadId: '1',
      payload: {
        mimeType: 'multipart/alternative',
        parts: [
          { mimeType: 'text/html', body: { data: b64url('<p>html</p>') } },
          { mimeType: 'text/plain', body: { data: b64url('plain text') } },
        ],
      },
    });
    expect(body).toBe('plain text');
  });
});

describe('drive', () => {
  it('escapes quotes in keywords', () => {
    expect(buildDriveQuery({ keywords: ["o'neil"] })).toContain("fullText contains 'o\\'neil'");
  });
});

describe('chunker', () => {
  it('splits long text into overlapping chunks under the size limit', () => {
    const text = Array.from({ length: 200 }, (_, i) => `Sentence number ${i}.`).join(' ');
    const chunks = splitText(text, 300, 50);
    expect(chunks.length).toBeGreaterThan(5);
    for (const c of chunks) expect(c.length).toBeLessThanOrEqual(300);
    expect(chunks.join(' ')).toContain('Sentence number 199.');
  });
});

describe('oauth ids', () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { reversedClientId, iosRedirectUri } = require('../src/google/oauthIds');
  it('reverses an iOS client id into its redirect scheme', () => {
    expect(reversedClientId('123-abc.apps.googleusercontent.com')).toBe('com.googleusercontent.apps.123-abc');
    expect(iosRedirectUri(' 123-abc.apps.googleusercontent.com ')).toBe('com.googleusercontent.apps.123-abc:/oauthredirect');
    expect(reversedClientId('')).toBeUndefined();
    expect(reversedClientId('not-a-client-id')).toBeUndefined();
  });
});
