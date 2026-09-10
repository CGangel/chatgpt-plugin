import { Config } from './config.js'

/**
 * 各家供应商思考(推理)参数的统一适配层，按模式独立配置，取值使用供应商原生原文
 *
 * API模式（OpenAI兼容请求体，reasoning_effort）：
 *   OpenAI    原生取值: none | low | medium | high | xhigh | max，直传
 *   DeepSeek  原生取值: none | low | high | max；medium、xhigh按官方兼容映射降为high
 *   格式配置 apiThinkingFormat: auto(按baseUrl识别deepseek) | openai | deepseek
 *
 * Gemini模式（仅谷歌原生协议，generationConfig.thinkingConfig，OpenAI兼容端点请使用API模式）：
 *   gemini-3系 thinkingLevel: low | high（Pro系列无法完全关闭）
 *   gemini-2.5系 thinkingBudget: 0关闭 / -1动态 / 具体数值
 *   统一配置 geminiThinkingLevel: off | low | high | dynamic
 *
 * Qwen（DashScope原生 parameters）：
 *   enable_thinking: true|false + thinking_budget: 数值
 *   统一配置 qwenThinking: off | on | 数字(1024~32768)
 *
 * GLM：仅适配glm-5.3/glm-5.3-flash及以上（思考强制启用，thinking.type仅enabled），
 *   强度用顶层 reasoning_effort: low | high | max（默认max）；4.5~5.2沿用thinking.type开关
 *
 * Claude：thinking: {type: enabled, budget_tokens: >=1024}，不传即为关闭
 */

// reasoning_effort 原生取值超集（openai全集，deepseek按映射归一化）
const EFFORTS = ['none', 'low', 'medium', 'high', 'xhigh', 'max']

/**
 * DeepSeek原生取值归一化：none/low/high/max直传，medium、xhigh按官方兼容映射降为high
 * @param {string} effort
 * @returns {string}
 */
export function deepseekEffort (effort) {
  return { none: 'none', low: 'low', medium: 'high', high: 'high', xhigh: 'high', max: 'max' }[effort]
}

/**
 * 解析API模式思考格式。auto按baseUrl识别：域名含deepseek则用DeepSeek取值
 * @param {string} baseUrl
 * @returns {'openai'|'deepseek'}
 */
export function resolveThinkingFormat (baseUrl) {
  const value = Config.apiThinkingFormat
  if (value === 'openai' || value === 'deepseek') {
    return value
  }
  return /deepseek/i.test(baseUrl || '') ? 'deepseek' : 'openai'
}

/**
 * 按格式归一化reasoning_effort取值：openai直传原文，deepseek映射到其原生取值集
 * @param {string} effort
 * @param {'openai'|'deepseek'} format
 * @returns {string}
 */
export function effortFor (effort, format) {
  return format === 'deepseek' ? deepseekEffort(effort) : effort
}

/**
 * 读取API模式思考强度（reasoning_effort原文），空串表示不发送
 * @returns {string}
 */
export function getApiThinkingEffort () {
  return EFFORTS.includes(Config.apiThinkingEffort) ? Config.apiThinkingEffort : ''
}

/**
 * 读取Gemini思考档位（off|low|high|dynamic），空串表示不发送
 * @returns {string}
 */
export function getGeminiThinkingLevel () {
  return ['off', 'low', 'high', 'dynamic'].includes(Config.geminiThinkingLevel) ? Config.geminiThinkingLevel : ''
}

/**
 * 读取Qwen思考设置：off关闭、on开启(默认预算)、或具体thinking_budget数值，空串表示不发送
 * @returns {string}
 */
export function getQwenThinking () {
  const value = Config.qwenThinking
  return (value === 'off' || value === 'on' || /^\d+$/.test(value)) ? value : ''
}

/**
 * 读取GLM思考强度原文（low|high|max），空串表示不发送。
 * 仅glm-5.3/glm-5.3-flash及以上生效（思考强制启用）；4.5~5.2沿用chatglmThinking开关
 * @returns {string}
 */
export function getChatglmThinkingEffort () {
  return ['low', 'high', 'max'].includes(Config.chatglmThinkingEffort) ? Config.chatglmThinkingEffort : ''
}

/**
 * 读取Claude思考预算budget_tokens（原生数字），0表示不发送（即关闭）
 * @returns {number}
 */
export function getClaudeThinkingBudget () {
  const value = Config.claudeThinkingBudget
  return /^\d+$/.test(value) ? parseInt(value) : 0
}

/**
 * Gemini官方API的thinkingConfig（谷歌原生格式）
 * @param {string} model 模型名
 * @param {'off'|'low'|'high'|'dynamic'} level
 * @returns {{thinkingBudget: number}|{thinkingLevel: string}|null} null表示不发送
 */
export function geminiThinkingConfig (model, level) {
  // gemini-3及以上使用thinkingLevel（仅low/high，Pro系列无法完全关闭思考）
  if (/gemini[-_]([3-9]|[1-9][0-9])/i.test(model || '')) {
    switch (level) {
      case 'low':
      case 'off': // 3系无法完全关闭，降级为low
        return { thinkingLevel: 'low' }
      case 'high':
        return { thinkingLevel: 'high' }
      case 'dynamic': // 3系默认即为动态思考，不发送
        return null
    }
  }
  // gemini-2.5系使用thinkingBudget：0关闭，-1动态
  return { thinkingBudget: { off: 0, low: 2048, high: 24576, dynamic: -1 }[level] }
}
