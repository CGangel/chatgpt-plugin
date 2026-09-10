import crypto from 'crypto'
import { GoogleGeminiClient } from './GoogleGeminiClient.js'
import { newFetch } from '../utils/proxy.js'
import { getGeminiThinkingLevel, geminiThinkingConfig } from '../utils/thinking.js'
import _ from 'lodash'

const BASEURL = 'https://generativelanguage.googleapis.com'

function normalizeBaseUrl (baseUrl) {
  return (baseUrl || BASEURL).replace(/\/+$/, '')
}

function buildGeminiGenerateContentUrl (baseUrl, model) {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl)
  return normalizedBaseUrl.endsWith('/v1') || normalizedBaseUrl.endsWith('/v1beta')
    ? `${normalizedBaseUrl}/models/${model}:generateContent`
    : `${normalizedBaseUrl}/v1beta/models/${model}:generateContent`
}

export const HarmCategory = {
  HARM_CATEGORY_UNSPECIFIED: 'HARM_CATEGORY_UNSPECIFIED',
  HARM_CATEGORY_HATE_SPEECH: 'HARM_CATEGORY_HATE_SPEECH',
  HARM_CATEGORY_SEXUALLY_EXPLICIT: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
  HARM_CATEGORY_HARASSMENT: 'HARM_CATEGORY_HARASSMENT',
  HARM_CATEGORY_DANGEROUS_CONTENT: 'HARM_CATEGORY_DANGEROUS_CONTENT',
  HARM_CATEGORY_CIVIC_INTEGRITY: 'HARM_CATEGORY_CIVIC_INTEGRITY'
}

export const HarmBlockThreshold = {
  HARM_BLOCK_THRESHOLD_UNSPECIFIED: 'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
  BLOCK_LOW_AND_ABOVE: 'BLOCK_LOW_AND_ABOVE',
  BLOCK_MEDIUM_AND_ABOVE: 'BLOCK_MEDIUM_AND_ABOVE',
  BLOCK_ONLY_HIGH: 'BLOCK_ONLY_HIGH',
  BLOCK_NONE: 'BLOCK_NONE',
  OFF: 'OFF'
}



export class CustomGoogleGeminiClient extends GoogleGeminiClient {
  constructor (props) {
    super(props)
    this.model = props.model
    this.baseUrl = normalizeBaseUrl(props.baseUrl)
    this.debug = props.debug
  }


  async sendMessage (text, opt = {}) {
    let history = await this.getHistory(opt.parentMessageId)
    let systemMessage = opt.system
    const idThis = crypto.randomUUID()
    const idModel = crypto.randomUUID()
    const thisMessage = {
      role: 'user',
      parts: text ? [{ text }] : [],
      id: idThis,
      parentMessageId: opt.parentMessageId || undefined
    }
    if (opt.image) {
      thisMessage.parts.push({
        inline_data: {
          mime_type: 'image/jpeg',
          data: opt.image
        }
      })
    }
    history.push(_.cloneDeep(thisMessage))

    let url
    let body
    let headers

    // --- 原生 Gemini API 逻辑（仅支持谷歌原生协议，OpenAI兼容端点请使用API模式） ---
    url = buildGeminiGenerateContentUrl(this.baseUrl, this.model)
    headers = {
      'x-goog-api-key': this._key
    }
    body = {
      contents: history,
      safetySettings: [
        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.OFF },
        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.OFF },
        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.OFF },
        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.OFF },
        { category: HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold: HarmBlockThreshold.BLOCK_NONE }
      ],
      generationConfig: {
        maxOutputTokens: opt.maxOutputTokens || 4096,
        temperature: opt.temperature || 0.9,
        topP: opt.topP || 0.95,
        topK: opt.tokK || 16
      },
      tools: []
    }
    if (systemMessage) {
      body.system_instruction = { parts: { text: systemMessage } }
    }
    // 思考强度（谷歌原生格式：3系thinkingLevel，2.5系thinkingBudget）
    const thinkingLevel = getGeminiThinkingLevel()
    if (thinkingLevel) {
      const thinkingConfig = geminiThinkingConfig(this.model, thinkingLevel)
      if (thinkingConfig) {
        body.generationConfig.thinkingConfig = thinkingConfig
      }
    }
    if (opt.search) {
      body.tools.push({ google_search: {} })
    }
    if (opt.codeExecution) {
      body.tools.push({ code_execution: {} })
    }
    if (opt.image) {
      delete body.tools
    }
    body.contents.forEach(content => {
      delete content.id
      delete content.parentMessageId
      delete content.conversationId
    })

    if (this.debug) {
      logger.debug(`Request Body to ${url}: ${JSON.stringify(body)}`)
    }

    let result
    try {
      result = await newFetch(url, {
        method: 'POST',
        body: JSON.stringify(body),
        headers
      })
    } catch (fetchErr) {
      logger.error(`[Gemini] 网络请求失败 - URL: ${url}`)
      logger.error(`[Gemini] 错误详情: ${fetchErr.message}`)
      if (fetchErr.code) logger.error(`[Gemini] 错误码: ${fetchErr.code}`)
      if (fetchErr.cause) logger.error(`[Gemini] 错误原因: ${JSON.stringify(fetchErr.cause)}`)
      throw fetchErr
    }

    if (result.status !== 200) {
      const errorBody = await result.text()
      logger.error(`[Gemini] API返回非200状态 - 状态码: ${result.status}, 响应体: ${errorBody}`)
      throw new Error(`API request failed with status ${result.status}: ${errorBody}`)
    }

    /** @type {Content | undefined} */
    let responseContent
    let groundingMetadata // 仅 Gemini 原生 API 支持

    let rawText
    try {
      rawText = await result.text()
    } catch (textErr) {
      logger.error(`[Gemini] 无法读取响应体: ${textErr.message}`)
      throw textErr
    }

    // --- 解析原生 Gemini 响应 ---
    /** @type {{candidates: Array<{content: Content, groundingMetadata: GroundingMetadata, finishReason: string}>}} */
    let response
    try {
      response = JSON.parse(rawText)
    } catch (parseErr) {
      logger.error(`[Gemini] 原生API响应JSON解析失败 - 原始响应: ${rawText}`)
      throw new Error(`Gemini native API returned non-JSON response: ${rawText.substring(0, 500)}`)
    }
    if (this.debug) {
      console.log('Gemini Response:', JSON.stringify(response))
    }
    if (!response.candidates || response.candidates.length === 0) {
      logger.error(`[Gemini] 原生API无candidates返回 - 完整响应: ${JSON.stringify(response)}`)
      // 无内容回复，可在此处添加重试逻辑
      throw new Error('Gemini API returned no candidates.')
    }
    responseContent = response.candidates[0].content
    groundingMetadata = response.candidates[0].groundingMetadata
    if (responseContent) {
      await this.upsertMessage(thisMessage)
      const respMessage = Object.assign(responseContent, { id: idModel, parentMessageId: idThis })
      await this.upsertMessage(respMessage)
    }
    let { final } = handleSearchResponse(responseContent)
    try {
      if (groundingMetadata?.groundingChunks) {
        final += '\n参考资料\n'
        groundingMetadata.groundingChunks.forEach(chunk => {
          final += `[${chunk.web.title}]\n`
        })
        groundingMetadata.webSearchQueries.forEach(q => {
          logger.info('search query: ' + q)
        })
      }
    } catch (err) {
      logger.warn(err)
    }

    return {
      text: final,
      conversationId: '',
      parentMessageId: idModel, //  parentMessageId 应该返回模型消息的 id，下一轮基于此回复
      id: idModel
    }
  }
}

/**
 * 处理成单独的text
 * @param {Content} responseContent
 * @returns {{final: string, responseContent}}
 */
function handleSearchResponse (responseContent) {
  let final = ''

  // 遍历每个 part 并处理
  responseContent.parts = responseContent.parts.map((part) => {
    let newText = ''

    if (part.text) {
      newText += part.text
      final += part.text // 累积到 final
    }
    if (part.executableCode) {
      const codeBlock = '\n执行代码：\n' + '```' + part.executableCode.language + '\n' + part.executableCode.code.trim() + '\n```\n\n'
      newText += codeBlock
      final += codeBlock // 累积到 final
    }
    if (part.codeExecutionResult) {
      const resultBlock = `\n执行结果(${part.codeExecutionResult.outcome})：\n` + '```\n' + part.codeExecutionResult.output + '\n```\n\n'
      newText += resultBlock
      final += resultBlock // 累积到 final
    }

    // 返回更新后的 part，但不设置空的 text
    const updatedPart = { ...part }
    if (newText) {
      updatedPart.text = newText // 仅在 newText 非空时设置 text
    } else {
      delete updatedPart.text // 如果 newText 是空的，则删除 text 字段
    }

    return updatedPart
  })

  return {
    final,
    responseContent
  }
}
