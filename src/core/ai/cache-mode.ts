/**
 * Per-provider prompt-cache mode overrides — the `cache_mode` config key.
 *
 * Some providers cache prompt prefixes automatically (OpenAI, DeepSeek,
 * Gemini, Groq, LiteLLM-proxied backends, …) while their recipe declares
 * `supports_prompt_cache: false`, because gbrain cannot inject the explicit
 * `cache_control` markers those providers would need. `classifyCapabilities`
 * then reports `degraded:no_caching`, so `gbrain doctor` and the subagent
 * model resolver print a cost-regression warning for a provider that is
 * already caching.
 *
 * `gbrain config set cache_mode '{"litellm":"auto"}'` is the operator escape
 * hatch: the named providers report `supportsPromptCaching: true`, which
 * silences that warning. It does NOT change what the gateway sends — cache
 * marker injection stays gated on the recipe's own `supports_prompt_cache`
 * field, so an Anthropic `cache_control` marker never leaks to a
 * non-Anthropic provider.
 *
 * Storage: ONE `cache_mode` row in the config table holding a JSON object
 * (`gbrain config set cache_mode '{"litellm":"auto"}'`). The column is
 * `TEXT`, so readers get a string — `parseCacheMode` owns the parse. Same
 * shape as `embedding_columns`. Pre-parsed objects are accepted too, for
 * tests and direct SDK callers.
 */
export type CacheMode = Record<string, 'auto'>;

/** The only engine surface this module needs (matches config-snapshot's ConfigReader). */
export interface CacheModeReader {
  getConfig(key: string): Promise<string | null | undefined>;
}

/**
 * Coerce a raw `cache_mode` value into a validated map.
 *
 * Accepts the DB-plane JSON string and an already-parsed object. Only
 * `'auto'` is a recognized value; malformed JSON, non-object payloads, and
 * unknown modes are dropped silently — this is a warning-suppression knob,
 * so a typo must degrade to "no override", never crash a think/subagent run.
 */
export function parseCacheMode(raw: unknown): CacheMode | undefined {
  if (raw === null || raw === undefined || raw === '') return undefined;
  let parsed: unknown = raw;
  if (typeof raw === 'string') {
    try {
      parsed = JSON.parse(raw);
    } catch {
      return undefined;
    }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined;
  const out: CacheMode = {};
  for (const [provider, mode] of Object.entries(parsed as Record<string, unknown>)) {
    if (mode === 'auto') out[provider] = 'auto';
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/**
 * Read + parse the `cache_mode` config row. Never throws: a missing key or a
 * read failure means "no override" and the recipe default stands.
 */
export async function readCacheMode(
  reader: CacheModeReader | null | undefined,
): Promise<CacheMode | undefined> {
  if (!reader) return undefined;
  try {
    return parseCacheMode(await reader.getConfig('cache_mode'));
  } catch {
    return undefined;
  }
}
