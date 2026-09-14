/**
 * File-plane reranker knobs — `reranker_model` / `reranker_enabled`.
 *
 * `embedding_model` and `chat_model` are declared in `~/.gbrain/config.json`,
 * but the reranker knobs were DB-only (`search.reranker.*`), so a
 * config.json-only install had no way to point rerank at a model: the DB read
 * silently ignored the file plane, and the only symptom was search quietly
 * running un-reranked. These tests pin the overlay and its precedence
 * (gbrain's usual order):
 *
 *   per-call > env > file (config.json) > DB (`search.reranker.*`) > mode bundle
 *
 * A LiteLLM proxy can front a rerank model, so `litellm:<model>` is the
 * motivating value here.
 */
import { describe, test, expect, beforeEach, afterEach, spyOn } from 'bun:test';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { loadSearchModeConfig, resolveSearchMode, type SearchPerCallOpts } from '../../src/core/search/mode.ts';
import type { BulkConfigReader } from '../../src/core/config-snapshot.ts';
import { withEnv } from '../helpers/with-env.ts';
import { runConfig } from '../../src/commands/config.ts';
import type { BrainEngine } from '../../src/core/engine.ts';

/** DB-plane stub: one row per key, no snapshot (exercises the per-key path). */
function dbReader(map: Record<string, string>): BulkConfigReader {
  return {
    getConfig: async (k: string) => map[k] ?? null,
  };
}

async function knobsOf(map: Record<string, string>, perCall?: SearchPerCallOpts) {
  return resolveSearchMode({ ...(await loadSearchModeConfig(dbReader(map))), perCall });
}

let home: string;
let priorHome: string | undefined;

/** Point GBRAIN_HOME at a fresh temp dir and write its config.json. */
function writeHomeConfig(obj: Record<string, unknown>): void {
  mkdirSync(join(home, '.gbrain'), { recursive: true });
  writeFileSync(
    join(home, '.gbrain', 'config.json'),
    JSON.stringify({ engine: 'pglite', ...obj }),
  );
  process.env.GBRAIN_HOME = home;
}

beforeEach(() => {
  priorHome = process.env.GBRAIN_HOME;
  home = mkdtempSync(join(tmpdir(), 'gbrain-reranker-file-plane-'));
});

afterEach(() => {
  process.env.GBRAIN_HOME = priorHome;
  rmSync(home, { recursive: true, force: true });
});

describe('reranker knobs — config.json / env support', () => {
  test('DB-only search.reranker.model still resolves (baseline, unchanged)', async () => {
    writeHomeConfig({});
    const k = await knobsOf({
      'search.reranker.model': 'voyage:rerank-2.5',
      'search.reranker.enabled': 'true',
    });
    expect(k.reranker_model).toBe('voyage:rerank-2.5');
    expect(k.reranker_enabled).toBe(true);
  });

  test('config.json reranker_model beats the DB row', async () => {
    writeHomeConfig({ reranker_model: 'litellm:rerank-zh' });
    const k = await knobsOf({ 'search.reranker.model': 'voyage:rerank-2.5' });
    expect(k.reranker_model).toBe('litellm:rerank-zh');
  });

  test('GBRAIN_RERANKER_MODEL env beats config.json', async () => {
    writeHomeConfig({ reranker_model: 'litellm:from-file' });
    await withEnv({ GBRAIN_RERANKER_MODEL: 'litellm:from-env' }, async () => {
      const k = await knobsOf({ 'search.reranker.model': 'voyage:rerank-2.5' });
      expect(k.reranker_model).toBe('litellm:from-env');
    });
  });

  test('config.json reranker_enabled:false wins over a DB true — no silent rerank', async () => {
    writeHomeConfig({ reranker_enabled: false });
    const k = await knobsOf({
      'search.reranker.enabled': 'true',
      'search.reranker.model': 'voyage:rerank-2.5',
    });
    expect(k.reranker_enabled).toBe(false);
  });

  test('config.json reranker_enabled:true wins over a DB false', async () => {
    writeHomeConfig({ reranker_enabled: true });
    const k = await knobsOf({ 'search.reranker.enabled': 'false' });
    expect(k.reranker_enabled).toBe(true);
  });

  test('per-call still beats every config plane', async () => {
    writeHomeConfig({ reranker_model: 'litellm:from-file' });
    const k = await knobsOf(
      { 'search.reranker.model': 'voyage:rerank-2.5' },
      { reranker_model: 'litellm:per-call' },
    );
    expect(k.reranker_model).toBe('litellm:per-call');
  });

  test('blank file-plane model falls through to the DB row (not a mute)', async () => {
    writeHomeConfig({ reranker_model: '   ' });
    const k = await knobsOf({ 'search.reranker.model': 'voyage:rerank-2.5' });
    expect(k.reranker_model).toBe('voyage:rerank-2.5');
  });

  test('absent config.json leaves DB + mode-bundle defaults untouched', async () => {
    // No writeHomeConfig(): configDir() resolves under the temp home with no
    // file, so the overlay contributes nothing.
    process.env.GBRAIN_HOME = home;
    const k = await knobsOf({ 'search.reranker.model': 'litellm:x' });
    expect(k.reranker_model).toBe('litellm:x');
  });
});

// ── CLI write path ────────────────────────────────────────────────────────
// `gbrain config set reranker_model ...` must land in the file plane, because
// that is where the overlay above reads. A DB row under this name would be
// echoed by `config get` and read by nothing — the silent-no-op class this
// repo has already paid for twice (cache_mode, self_upgrade.mode). The last
// test here is the load-bearing one: it proves the write reaches the reader.
describe('reranker knobs — `config set` routes to the file plane', () => {
  function setStubEngine(): { engine: BrainEngine; setCalls: Array<[string, string]> } {
    const setCalls: Array<[string, string]> = [];
    const engine = {
      getConfig: async () => null,
      setConfig: async (key: string, value: string) => { setCalls.push([key, value]); },
    } as unknown as BrainEngine;
    return { engine, setCalls };
  }

  async function runConfigCapture(
    engine: BrainEngine,
    args: string[],
  ): Promise<{ logs: string[]; errs: string[]; exit: number | null }> {
    const logs: string[] = [];
    const errs: string[] = [];
    let exit: number | null = null;
    const logSpy = spyOn(console, 'log').mockImplementation((...a: unknown[]) => { logs.push(a.join(' ')); });
    const errSpy = spyOn(console, 'error').mockImplementation((...a: unknown[]) => { errs.push(a.join(' ')); });
    const exitSpy = spyOn(process, 'exit').mockImplementation(((code?: number) => {
      exit = code ?? 0;
      throw new Error(`EXIT:${code}`);
    }) as never);
    try {
      await runConfig(engine, args);
    } catch (e) {
      if (!(e as Error).message.startsWith('EXIT:')) throw e;
    } finally {
      logSpy.mockRestore();
      errSpy.mockRestore();
      exitSpy.mockRestore();
    }
    return { logs, errs, exit };
  }

  const homeConfig = (): Record<string, unknown> =>
    JSON.parse(readFileSync(join(home, '.gbrain', 'config.json'), 'utf8'));

  test('reranker_model is written to config.json, NOT the DB', async () => {
    writeHomeConfig({});
    const { engine, setCalls } = setStubEngine();
    const { exit, errs } = await runConfigCapture(engine, ['set', 'reranker_model', 'litellm:rerank-zh']);
    expect(exit).toBeNull();
    expect(errs.join('\n')).not.toContain('Nothing was written');
    expect(setCalls).toEqual([]); // no DB row
    expect(homeConfig().reranker_model).toBe('litellm:rerank-zh');
  });

  test('a prefix-less model is refused and nothing is written', async () => {
    writeHomeConfig({});
    const { engine, setCalls } = setStubEngine();
    const { exit, errs } = await runConfigCapture(engine, ['set', 'reranker_model', 'rerank-zh']);
    expect(exit).toBe(1);
    expect(errs.join('\n')).toContain('<provider>:<model>');
    expect(setCalls).toEqual([]);
    expect(homeConfig().reranker_model).toBeUndefined();
  });

  test('reranker_enabled parses to a real boolean in config.json', async () => {
    writeHomeConfig({});
    const { engine } = setStubEngine();
    await runConfigCapture(engine, ['set', 'reranker_enabled', 'false']);
    expect(homeConfig().reranker_enabled).toBe(false);
    await runConfigCapture(engine, ['set', 'reranker_enabled', 'yes']);
    expect(homeConfig().reranker_enabled).toBe(true);
  });

  test('unset removes the file-plane key', async () => {
    writeHomeConfig({ reranker_model: 'litellm:rerank-zh' });
    const { engine } = setStubEngine();
    const { exit } = await runConfigCapture(engine, ['unset', 'reranker_model']);
    expect(exit).toBeNull();
    expect(homeConfig().reranker_model).toBeUndefined();
  });

  test('the CLI write actually reaches the reranker resolver (no silent no-op)', async () => {
    writeHomeConfig({});
    const { engine } = setStubEngine();
    await runConfigCapture(engine, ['set', 'reranker_model', 'litellm:rerank-zh']);
    const k = await knobsOf({ 'search.reranker.model': 'voyage:rerank-2.5' });
    expect(k.reranker_model).toBe('litellm:rerank-zh');
  });
});
