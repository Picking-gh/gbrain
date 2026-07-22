/**
 * Pins `cache_mode` parsing.
 *
 * The key is stored as a JSON object in a TEXT config column, so every real
 * reader gets a STRING back. The first cut of this feature read the value as an
 * object (`typeof raw === 'object'`) and therefore no-op'd on every real
 * engine while passing against an object-returning test stub — these tests use
 * the string shape for that reason.
 */
import { describe, test, expect } from 'bun:test';
import { parseCacheMode, readCacheMode } from '../../src/core/ai/cache-mode.ts';

describe('parseCacheMode', () => {
  test('parses the DB-plane JSON string (the real storage shape)', () => {
    expect(parseCacheMode('{"litellm":"auto"}')).toEqual({ litellm: 'auto' });
    expect(parseCacheMode('{"litellm":"auto","openai":"auto"}'))
      .toEqual({ litellm: 'auto', openai: 'auto' });
  });

  test('accepts an already-parsed object (tests / direct SDK callers)', () => {
    expect(parseCacheMode({ openai: 'auto' })).toEqual({ openai: 'auto' });
  });

  test('drops unrecognized modes and keeps the recognized ones', () => {
    expect(parseCacheMode('{"openai":"auto","groq":"always","x":"AUTO"}'))
      .toEqual({ openai: 'auto' });
  });

  test('no recognized providers left → undefined (recipe default stands)', () => {
    expect(parseCacheMode('{"openai":"always"}')).toBeUndefined();
  });

  test('malformed / empty / non-object payloads → undefined, never a throw', () => {
    for (const bad of ['not json', '', null, undefined, '[]', '[{"a":"auto"}]', '"auto"', '42', 'true']) {
      expect(parseCacheMode(bad)).toBeUndefined();
    }
  });
});

describe('readCacheMode', () => {
  const readerOf = (value: unknown) => ({
    getConfig: async () => value as string | null | undefined,
  });

  test('parses the string a real engine returns for the row', async () => {
    expect(await readCacheMode(readerOf('{"litellm":"auto"}'))).toEqual({ litellm: 'auto' });
  });

  test('missing row → undefined', async () => {
    expect(await readCacheMode(readerOf(null))).toBeUndefined();
    expect(await readCacheMode(readerOf(undefined))).toBeUndefined();
    expect(await readCacheMode(readerOf(''))).toBeUndefined();
  });

  test('a read failure degrades to no override instead of throwing', async () => {
    const throwing = { getConfig: async (): Promise<string | null> => { throw new Error('db down'); } };
    expect(await readCacheMode(throwing)).toBeUndefined();
  });

  test('no reader → undefined', async () => {
    expect(await readCacheMode(null)).toBeUndefined();
    expect(await readCacheMode(undefined)).toBeUndefined();
  });
});
