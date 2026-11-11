/**
 * Web APIs that LangGraph.js expects and Hermes (the React Native JS engine) may lack.
 * Imported first in index.ts, before anything loads LangGraph. Each one is installed only if missing.
 *
 * - ReadableStream & co: @langchain/core's stream utilities extend ReadableStream at load time.
 * - TextEncoder / TextDecoder: the checkpoint serializer encodes state as UTF-8 bytes.
 * - crypto.getRandomValues / randomUUID: checkpoint and run ids are UUIDs.
 * - Symbol.asyncIterator: LangGraph streams are consumed with `for await`. Older Hermes lacks the symbol.
 * - URL: React Native's built-in URL throws on most getters; @langchain/core's JSON-schema code
 *   builds a URL at load time and reads its fields later. This installs a spec-compliant one.
 */
import 'react-native-url-polyfill/auto';
import * as ExpoCrypto from 'expo-crypto';
import { ReadableStream, TransformStream, WritableStream } from 'web-streams-polyfill';
import { utf8Decode, utf8Encode } from './core/encoding';

const g = globalThis as Record<string, unknown>;

if (typeof Symbol.asyncIterator === 'undefined') {
  // Must exist before LangGraph's classes are defined, so they get the right method key.
  (Symbol as unknown as Record<string, symbol>).asyncIterator = Symbol.for('Symbol.asyncIterator');
}

if (typeof g.ReadableStream === 'undefined') g.ReadableStream = ReadableStream;
if (typeof g.WritableStream === 'undefined') g.WritableStream = WritableStream;
if (typeof g.TransformStream === 'undefined') g.TransformStream = TransformStream;

if (typeof g.TextEncoder === 'undefined') {
  g.TextEncoder = class {
    readonly encoding = 'utf-8';
    encode(s = ''): Uint8Array {
      return utf8Encode(s);
    }
  };
}

if (typeof g.TextDecoder === 'undefined') {
  g.TextDecoder = class {
    readonly encoding = 'utf-8';
    decode(input?: ArrayBuffer | ArrayBufferView): string {
      if (!input) return '';
      const bytes =
        input instanceof Uint8Array
          ? input
          : ArrayBuffer.isView(input)
            ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
            : new Uint8Array(input);
      return utf8Decode(bytes);
    }
  };
}

const c = (g.crypto ?? {}) as Record<string, unknown>;
if (typeof c.getRandomValues !== 'function') {
  c.getRandomValues = <T extends ArrayBufferView>(arr: T): T =>
    ExpoCrypto.getRandomValues(arr as unknown as Uint8Array) as unknown as T;
}
if (typeof c.randomUUID !== 'function') c.randomUUID = () => ExpoCrypto.randomUUID();
g.crypto = c;
