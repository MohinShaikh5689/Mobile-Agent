import { base64ToBytes, bytesToBase64, utf8Decode, utf8Encode } from '../src/core/encoding';

describe('polyfills', () => {
  it('installs TextEncoder/TextDecoder and web streams when the engine lacks them', () => {
    const g = globalThis as Record<string, unknown>;
    const saved = { TE: g.TextEncoder, TD: g.TextDecoder, RS: g.ReadableStream };
    g.TextEncoder = undefined;
    g.TextDecoder = undefined;
    g.ReadableStream = undefined;
    try {
      jest.isolateModules(() => {
        require('../src/polyfills');
      });
      const enc = new (g.TextEncoder as { new (): { encode(s: string): Uint8Array } })();
      const dec = new (g.TextDecoder as { new (): { decode(b: Uint8Array): string } })();
      expect(dec.decode(enc.encode('héllo ✓ 🎉'))).toBe('héllo ✓ 🎉');
      expect(typeof g.ReadableStream).toBe('function');
    } finally {
      g.TextEncoder = saved.TE;
      g.TextDecoder = saved.TD;
      g.ReadableStream = saved.RS;
    }
  });
});

describe('encoding helpers', () => {
  it('round-trips UTF-8 and base64', () => {
    const s = 'Visa — überprüft ✓ 日本 🎉';
    expect(utf8Decode(utf8Encode(s))).toBe(s);
    for (const n of [0, 1, 2, 3, 4, 5, 100]) {
      const bytes = Uint8Array.from({ length: n }, (_, i) => (i * 37) % 256);
      expect(Array.from(base64ToBytes(bytesToBase64(bytes)))).toEqual(Array.from(bytes));
    }
  });
});

describe('navigator.userAgent', () => {
  it("gives React Native's navigator a userAgent so LangChain's environment check does not throw", () => {
    const g = globalThis as Record<string, unknown>;
    const saved = { nav: g.navigator, win: g.window };
    // What React Native provides: a window global and a navigator without userAgent.
    g.window = g;
    g.navigator = { product: 'ReactNative' };
    try {
      jest.isolateModules(() => {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const env = require('@langchain/core/utils/env');
        expect(() => env.isJsDom()).toThrow(/includes/);
        require('../src/polyfills');
        expect((g.navigator as { userAgent?: string }).userAgent).toBe('ReactNative');
        expect(env.isJsDom()).toBe(false);
      });
    } finally {
      g.navigator = saved.nav;
      g.window = saved.win;
    }
  });
});
