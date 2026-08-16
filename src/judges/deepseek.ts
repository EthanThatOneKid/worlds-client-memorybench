import type { JudgeConfig } from "../types/judge"
import { DEFAULT_JUDGE_MODELS } from "../utils/models"
import { OpenAIJudge } from "./openai"

/**
 * DeepSeekJudge reuses the OpenAI judge implementation against DeepSeek's
 * OpenAI-compatible API (DEEPSEEK_BASE_URL), with its own provider name and
 * default model. JSON output mode is supported by DeepSeek, so the shared
 * JSON-scoring prompt parsing applies unchanged.
 */
export class DeepSeekJudge extends OpenAIJudge {
  name = "deepseek"

  override async initialize(config: JudgeConfig): Promise<void> {
    await super.initialize({
      ...config,
      baseUrl: config.baseUrl || "https://api.deepseek.com",
      model: config.model || DEFAULT_JUDGE_MODELS.deepseek,
    })
  }

  // deepseek-v4-flash is a reasoning model; disable thinking for judging so
  // the JSON score response is fast and the output budget is not consumed by
  // reasoning tokens.
  protected override getProviderOptions(): Record<string, unknown> {
    return { openai: { thinking: { type: "disabled" } } }
  }
}

export default DeepSeekJudge
