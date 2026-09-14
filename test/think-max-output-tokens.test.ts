/**
 * Pins `maxOutputTokensFor` — the per-model output-token budget `runThink`
 * passes to `client.create`. Thinking-by-default Claude 5 models
 * (`anthropic:claude-*-5`), OpenAI reasoning models (gpt-5 family,
 * o-series), and the Anthropic Claude 4.x deep-tier family (gbrain#4375 —
 * the default `deep` alias is anthropic:claude-opus-4-7, whose JSON envelope
 * truncated at 4000) get 16000; everything else stays 4000.
 *
 * Also pins the gbrain#4375 truncation labeling: a max_tokens-cut envelope
 * is 'output_truncated' / LLM_OUTPUT_TRUNCATED, never the generic not_json.
 *
 * When `think.max_output_tokens` config is set, the override wins regardless
 * of model — it becomes the caller's responsibility to budget appropriately.
 */
import { afterAll, beforeAll, beforeEach, describe, test, expect } from 'bun:test';
import { maxOutputTokensFor, runThink, type ThinkLLMClient } from '../src/core/think/index.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { importFromContent } from '../src/core/import-file.ts';

describe('maxOutputTokensFor — thinking-default headroom', () => {
  test('Claude 5 family gets 16000', () => {
    expect(maxOutputTokensFor('anthropic:claude-sonnet-5')).toBe(16000);
    expect(maxOutputTokensFor('anthropic:claude-opus-5')).toBe(16000);
    expect(maxOutputTokensFor('anthropic:claude-fable-5')).toBe(16000);
    expect(maxOutputTokensFor('anthropic:claude-haiku-5')).toBe(16000);
    expect(maxOutputTokensFor('anthropic/claude-sonnet-5')).toBe(16000); // slash form
  });

  test('OpenAI reasoning models (gpt-5 family, o-series) get 16000', () => {
    expect(maxOutputTokensFor('openai:gpt-5')).toBe(16000);
    expect(maxOutputTokensFor('openai:gpt-5.2')).toBe(16000);
    expect(maxOutputTokensFor('openai:gpt-5.5')).toBe(16000);
    expect(maxOutputTokensFor('openai:gpt-5-mini')).toBe(16000);
    expect(maxOutputTokensFor('openai:o1')).toBe(16000);
    expect(maxOutputTokensFor('openai:o3')).toBe(16000);
    expect(maxOutputTokensFor('openai:o4-mini')).toBe(16000);
    expect(maxOutputTokensFor('openai/gpt-5.2')).toBe(16000); // slash form
  });

  test('gbrain#4375 — Anthropic Claude 4.x family gets 16000 (deep-tier truncation headroom)', () => {
    // The default deep tier (model-config.ts) is anthropic:claude-opus-4-7:
    // at 4000 the synthesis JSON envelope truncated mid-stream and was
    // mislabeled not_json. Anthropic-scoped only, so provider hard caps
    // (DeepSeek 8192, gpt-4o) are unaffected.
    expect(maxOutputTokensFor('anthropic:claude-opus-4-7')).toBe(16000);
    expect(maxOutputTokensFor('anthropic:claude-opus-4-8')).toBe(16000);
    expect(maxOutputTokensFor('anthropic:claude-haiku-4-5')).toBe(16000);
    expect(maxOutputTokensFor('anthropic:claude-sonnet-4-6')).toBe(16000);
    expect(maxOutputTokensFor('anthropic/claude-opus-4-7')).toBe(16000); // slash form
    expect(maxOutputTokensFor('claude-opus-4-7')).toBe(16000);           // bare spelling
    expect(maxOutputTokensFor('openrouter:anthropic/claude-opus-4-7')).toBe(16000);
  });

  test('non-headroom models keep 4000', () => {
    // 3.x-era Claude spellings never match the 4.x family regex.
    expect(maxOutputTokensFor('anthropic:claude-3-haiku')).toBe(4000);
    expect(maxOutputTokensFor('anthropic:claude-3-5-sonnet')).toBe(4000);
    expect(maxOutputTokensFor('openai:gpt-4o')).toBe(4000);
    expect(maxOutputTokensFor('openai:gpt-4o-mini')).toBe(4000);
    expect(maxOutputTokensFor('openai:gpt-4.1')).toBe(4000);
    // Non-reasoning ChatGPT snapshots of the gpt-5 family stay at 4000.
    expect(maxOutputTokensFor('openai:gpt-5-chat-latest')).toBe(4000);
    expect(maxOutputTokensFor('openai:gpt-5.2-chat-latest')).toBe(4000);
    // Scope is the gpt-5 family + numbered o-series only — other OpenAI
    // reasoning-capable ids (e.g. codex-mini-latest) keep the conservative
    // default until deliberately added.
    expect(maxOutputTokensFor('openai:codex-mini-latest')).toBe(4000);
    // Version/name boundaries: `gpt-50` and `o3foo` are not gpt-5 / o3.
    expect(maxOutputTokensFor('openai:gpt-50')).toBe(4000);
    expect(maxOutputTokensFor('openai:o3foo')).toBe(4000);
    // Prefix must be the openai provider — a bare model name or another
    // provider's gpt-5 spelling doesn't match.
    expect(maxOutputTokensFor('o3')).toBe(4000);
    expect(maxOutputTokensFor('gpt-5.2')).toBe(4000);
    expect(maxOutputTokensFor('openrouter:openai/gpt-5.2')).toBe(4000);
  });

  test('gbrain#4172 — recipe-declared thinking-by-default models (DeepSeek v4) get 16000 via the capability layer', () => {
    // DeepSeek v4 thinks by default and bills reasoning against max_tokens:
    // at 4000 the whole budget is spent reasoning and think returns
    // truncated/empty JSON. Keyed on thinking_by_default (capability), not a
    // model-name regex, so provider renames keep the headroom.
    expect(maxOutputTokensFor('deepseek:deepseek-v4-flash')).toBe(16000);
    expect(maxOutputTokensFor('deepseek:deepseek-v4-pro')).toBe(16000);
    // OpenRouter DeepSeek hosts think by default too (#4758) — same recipe
    // capability, so the OR route gets the same headroom as native deepseek:.
    expect(maxOutputTokensFor('openrouter:deepseek/deepseek-v4-flash')).toBe(16000);
    expect(maxOutputTokensFor('openrouter:deepseek/deepseek-v4-flash-0731')).toBe(16000);
    // Retired alias still routes to a thinking v4 model at the provider.
    expect(maxOutputTokensFor('deepseek:deepseek-reasoner')).toBe(16000);
    // Recipes without the capability keep the conservative default.
    expect(maxOutputTokensFor('ollama:llama3.3')).toBe(4000);
    expect(maxOutputTokensFor('groq:llama-3.3-70b-versatile')).toBe(4000);
    // Unknown provider strings fail open to the default, never throw.
    expect(maxOutputTokensFor('nonexistent-provider:whatever')).toBe(4000);
    expect(maxOutputTokensFor('voyage:voyage-4')).toBe(4000); // chat-less recipe
  });

  test('gbrain#4727 — Zhipu GLM-4.5+/5.x get 16000 via the capability layer; older GLM ids keep 4000', () => {
    // GLM-4.5+ and the GLM-5.x series reason by default and bill reasoning
    // against max_tokens, same as DeepSeek v4: at 4000 the whole budget is
    // spent reasoning and think returns truncated/empty JSON.
    expect(maxOutputTokensFor('zhipu:glm-5.3-flash')).toBe(16000);
    expect(maxOutputTokensFor('zhipu:glm-5.3')).toBe(16000);
    expect(maxOutputTokensFor('zhipu:glm-5.1')).toBe(16000);
    expect(maxOutputTokensFor('zhipu:glm-4.6')).toBe(16000);
    expect(maxOutputTokensFor('zhipu:glm-4.5')).toBe(16000);
    // Pre-4.5 ids don't reason by default — conservative cap stands.
    expect(maxOutputTokensFor('zhipu:glm-4')).toBe(4000);
    expect(maxOutputTokensFor('zhipu:glm-4-plus')).toBe(4000);
    expect(maxOutputTokensFor('zhipu:glm-3-turbo')).toBe(4000);
  });

  test('override wins regardless of model', () => {
    expect(maxOutputTokensFor('anthropic:claude-sonnet-5', 8192)).toBe(8192);
    expect(maxOutputTokensFor('anthropic:claude-opus-4-8', 4096)).toBe(4096);
    expect(maxOutputTokensFor('openai:gpt-4o', 10000)).toBe(10000);
  });

  test('override undefined falls through to defaults', () => {
    expect(maxOutputTokensFor('anthropic:claude-sonnet-5', undefined)).toBe(16000);
    expect(maxOutputTokensFor('openai:gpt-4o', undefined)).toBe(4000);
  });
});

describe('runThink — max_tokens truncation labeling (gbrain#4375)', () => {
  let engine: PGLiteEngine;

  beforeAll(async () => {
    engine = new PGLiteEngine();
    await engine.connect({});
    await engine.initSchema();
    const imported = await importFromContent(
      engine,
      'notes/quokka-payments',
      '---\ntitle: Quokka Payments\ntype: note\n---\n\nThe quokka payments migration finished in March with zero downtime.',
      { noEmbed: true, sourceId: 'default' },
    );
    expect(imported.status).toBe('imported');
  });

  afterAll(async () => {
    await engine.disconnect();
  });

  // Same stub-message shape as auto-think-phase.test.ts so contextual typing
  // against the Anthropic Message union holds without casts.
  function clientOf(text: string, stopReason: 'end_turn' | 'max_tokens'): ThinkLLMClient {
    return {
      create: async () => ({
        id: 'msg_stub',
        type: 'message',
        role: 'assistant',
        model: 'anthropic:claude-opus-4-7',
        stop_reason: stopReason,
        stop_sequence: null,
        usage: { input_tokens: 900, output_tokens: 4000, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, server_tool_use: null, service_tier: null },
        content: [{ type: 'text', text }],
      }),
    };
  }

  test('stop_reason max_tokens + unparseable envelope → output_truncated, not the generic not_json', async () => {
    const result = await runThink(engine, {
      question: 'quokka payments migration status',
      // A max_tokens cut ends the JSON envelope mid-string — unparseable.
      client: clientOf('{"answer":"The quokka payments migration finished in Mar', 'max_tokens'),
      withTrajectory: false,
    });
    expect(result.warnings).toContain('LLM_OUTPUT_TRUNCATED');
    expect(result.synthesis_status).toBe('output_truncated');
    expect(result.warnings).not.toContain('LLM_OUTPUT_NOT_JSON');
    expect(result.synthesisOk).toBe(false);
    // Extractive-fallback contract holds for truncation like any other
    // compose failure: a non-empty gather still yields digest material.
    expect(result.pagesGathered).toBeGreaterThanOrEqual(1);
    expect(result.extractive).toBeDefined();
  });

  test('non-JSON output WITHOUT truncation keeps the generic not_json labeling', async () => {
    const result = await runThink(engine, {
      question: 'quokka payments migration status',
      client: clientOf('I cannot help with that request.', 'end_turn'),
      withTrajectory: false,
    });
    expect(result.synthesis_status).toBe('not_json');
    expect(result.warnings).toContain('LLM_OUTPUT_NOT_JSON');
    expect(result.warnings).not.toContain('LLM_OUTPUT_TRUNCATED');
  });
});

/**
 * The config read path itself (`readThinkMaxOutputTokens`), which the pure
 * `maxOutputTokensFor` tests above cannot cover: they prove the override
 * argument wins, not that `runThink` ever reads `think.max_output_tokens` or
 * wires it into `client.create`. This is the seam that would silently rot if
 * the config lookup broke — the pure test would stay green.
 *
 * Values are written as STRINGS because that is what the config table stores
 * (`value TEXT`); a reader that only handled numbers would pass a stub and
 * no-op on a real engine.
 */
describe('runThink — think.max_output_tokens config override', () => {
  let engine: PGLiteEngine;
  let captured: number[];

  beforeAll(async () => {
    engine = new PGLiteEngine();
    await engine.connect({});
    await engine.initSchema();
    const imported = await importFromContent(
      engine,
      'notes/quokka-payments',
      '---\ntitle: Quokka Payments\ntype: note\n---\n\nThe quokka payments migration finished in March with zero downtime.',
      { noEmbed: true, sourceId: 'default' },
    );
    expect(imported.status).toBe('imported');
    // Pin the model so the per-model default the override competes with is
    // deterministic (avoid depending on which provider keys the test shell has).
    await engine.setConfig('models.think', 'anthropic:claude-opus-4-7');
  });

  afterAll(async () => {
    await engine.disconnect();
  });

  beforeEach(async () => {
    captured = [];
    await engine.unsetConfig('think.max_output_tokens');
  });

  function capturingClient(): ThinkLLMClient {
    return {
      create: async (params) => {
        captured.push(params.max_tokens);
        return {
          id: 'msg_stub',
          type: 'message',
          role: 'assistant',
          model: 'anthropic:claude-opus-4-7',
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: { input_tokens: 10, output_tokens: 10, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, server_tool_use: null, service_tier: null },
          content: [{ type: 'text', text: '{"answer":"ok","citations":[]}' }],
        };
      },
    };
  }

  async function runOnce(): Promise<void> {
    await runThink(engine, {
      question: 'quokka payments migration status',
      client: capturingClient(),
      withTrajectory: false,
    });
  }

  test('unset → the per-model default reaches client.create (opus 4.x = 16000)', async () => {
    await runOnce();
    expect(captured).toEqual([16000]);
  });

  test('config value wins over the per-model default', async () => {
    await engine.setConfig('think.max_output_tokens', '8192');
    await runOnce();
    expect(captured).toEqual([8192]);
  });

  test('a value ABOVE the default also wins (raise the cap, not just clamp down)', async () => {
    await engine.setConfig('think.max_output_tokens', '32000');
    await runOnce();
    expect(captured).toEqual([32000]);
  });

  test('non-positive / unparseable values fall through to the default', async () => {
    for (const bad of ['0', '-5', 'abc', '', '   ']) {
      captured = [];
      await engine.setConfig('think.max_output_tokens', bad);
      await runOnce();
      expect(captured, `value=${JSON.stringify(bad)}`).toEqual([16000]);
    }
  });
});
