import { Config } from './config.js'

/**
 * 各家供应商思考(推理)参数的统一适配层
 *
 * Config.thinkingIntensity: 'default' | 'off' | 'low' | 'medium' | 'high'
 *   - default: 不发送任何思考参数，保持各供应商默认行为
 * Config.thinkingFormat: 'auto' | 'openai' | 'deepseek'
 *   - 仅影响API模式与gemini反代路径（OpenAI兼容请求体）
 *   - auto: 按baseUrl判断，域名含deepseek则用deepseek取值，否则用openai取值
 *
 * 各供应商取值：
 *   OpenAI    reasoning_effort: minimal|low|medium|high（off→minimal）
 *   DeepSeek  reasoning_effort: none|low|high|max（off→none）
 *   Gemini    原生 generationConfig.thinkingConfig:
 *             2.5系 thinkingBudget: 0关/2048/8192/24576，3系 thinkingLevel: low|high（无法完全关闭）
 *   Qwen      parameters.enable_thinking + thinking_budget: 1024/8192/32768
 *   GLM       thinking: {type: enabled|disabled}，无强度分级
 *   Claude    thinking: {type: enabled, budget_tokens: 1024/4096/16384}，不传即为关闭
 */

const INTENSITIES = ['off', 'low', 'medium', 'high']

/**
 * 读取思考强度，非法值回退为default
 * @returns {'default'|'off'|'low'|'medium'|'high'}
 */
export function getThinkingIntensity () {
  return INTENSITIES.includes(Config.thinkingIntensity) ? Config.thinkingIntensity : 'default'
}

/**
 * 解析OpenAI兼容请求应使用的思考参数取值风格
 * @param {string} baseUrl API地址，用于auto模式下识别deepseek
 * @returns {'openai'|'deepseek'}
 */
export function resolveThinkingFormat (baseUrl) {
  if (Config.thinkingFormat === 'openai' || Config.thinkingFormat === 'deepseek') {
    return Config.thinkingFormat
  }
  return /deepseek/i.test(baseUrl || '') ? 'deepseek' : 'openai'
}

/**
 * OpenAI取值：reasoning_effort minimal|low|medium|high
 * @param {'off'|'low'|'medium'|'high'} intensity
 * @returns {string}
 */
export function openaiReasoningEffort (intensity) {
  return { off: 'minimal', low: 'low', medium: 'medium', high: 'high' }[intensity]
}

/**
 * DeepSeek取值：reasoning_effort none|low|high|max
 * @param {'off'|'low'|'medium'|'high'} intensity
 * @returns {string}
 */
export function deepseekReasoningEffort (intensity) {
  return { off: 'none', low: 'low', medium: 'high', high: 'max' }[intensity]
}

/**
 * OpenAI兼容请求体（API模式/gemini反代）的reasoning_effort取值
 * @param {'off'|'low'|'medium'|'high'} intensity
 * @param {'openai'|'deepseek'} format
 * @returns {string}
 */
export function reasoningEffortFor (intensity, format) {
  return format === 'deepseek' ? deepseekReasoningEffort(intensity) : openaiReasoningEffort(intensity)
}

/**
 * Gemini原生API的thinkingConfig
 * @param {string} model 模型名
 * @param {'off'|'low'|'medium'|'high'} intensity
 * @returns {{thinkingBudget: number}|{thinkingLevel: string}|null}
 */
export function geminiThinkingConfig (model, intensity) {
  // gemini-3及以上使用thinkingLevel（仅low/high，Pro系列无法完全关闭思考）
  if (/gemini[-_]([3-9]|[1-9][0-9])/i.test(model || '')) {
    return { thinkingLevel: (intensity === 'off' || intensity === 'low') ? 'low' : 'high' }
  }
  // gemini-2.5系使用thinkingBudget，0为关闭
  return { thinkingBudget: { off: 0, low: 2048, medium: 8192, high: 24576 }[intensity] }
}
