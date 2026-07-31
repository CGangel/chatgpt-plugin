import crypto from 'crypto'
import { GoogleGeminiClient } from './GoogleGeminiClient.js'
import { newFetch } from '../utils/proxy.js'
import _ from 'lodash'

const BASEURL = 'https://generativelanguage.googleapis.com'
const OFFICIAL_GEMINI_HOST = 'generativelanguage.googleapis.com'

function normalizeBaseUrl (baseUrl) {
  return (baseUrl || BASEURL).replace(/\/+$/, '')
}

function isOfficialGeminiUrl (baseUrl) {
  try {
    return new URL(baseUrl).hostname === OFFICIAL_GEMINI_HOST
  } catch (err) {
    return normalizeBaseUrl(baseUrl).startsWith(BASEURL)
  }
}

function buildOpenAIChatCompletionsUrl (baseUrl) {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl)
  return normalizedBaseUrl.endsWith('/v1')
    ? `${normalizedBaseUrl}/chat/completions`
    : `${normalizedBaseUrl}/v1/chat/completions`
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
    this.supportFunction = true
    this.debug = props.debug
  }


  async sendMessage (text, opt = {}, retryTime = 3) {
    const isProxy = !isOfficialGeminiUrl(this.baseUrl)

    let history = await this.getHistory(opt.parentMessageId)
    let systemMessage = opt.system
    const idThis = crypto.randomUUID()
    const idModel = crypto.randomUUID()
    if (opt.functionResponse && !Array.isArray(opt.functionResponse)) {
      opt.functionResponse = [opt.functionResponse]
    }
    const thisMessage = opt.functionResponse?.length > 0
      ? {
          role: 'user',
          parts: opt.functionResponse.map(i => {
            return {
              functionResponse: i
            }
          }),
          id: idThis,
          parentMessageId: opt.parentMessageId || undefined
        }
      : {
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

    if (isProxy) {
      // --- 代理 API 逻辑 (OpenAI 格式) ---
      url = buildOpenAIChatCompletionsUrl(this.baseUrl)
      headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this._key}`
      }

      const messages = []
      // 1. 系统指令
      if (systemMessage) {
        messages.push({ role: 'system', content: systemMessage })
      }

      // 2. 转换历史记录和当前消息
      for (const geminiMsg of history) {
        const role = geminiMsg.role === 'model' ? 'assistant' : 'user'
        const content = convertGeminiPartsToOpenAIContent(geminiMsg.parts)
        if (content) { // 避免添加空内容的消息
          messages.push({ role, content })
        }
      }

      // 3. 构建请求体
      body = {
        model: this.model,
        messages,
        temperature: opt.temperature || 0.9,
        max_tokens: opt.maxOutputTokens || 4096,
        top_p: opt.topP || 0.95
      }

      // 4. 工具 (Function Calling)
      if (this.tools?.length > 0 && !opt.image) {
        body.tools = this.tools.map(tool => ({
          type: 'function',
          function: tool.function()
        }))
        // 映射 toolMode 到 tool_choice
        if (opt.toolMode && opt.toolMode !== 'AUTO') {
          body.tool_choice = opt.toolMode === 'NONE' ? 'none' : 'auto'
        }
      }
    } else {
      // --- 原生 Gemini API 逻辑 ---
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
      if (this.tools?.length > 0) {
        body.tools.push({ function_declarations: this.tools.map(tool => tool.function()) })
        let mode = opt.toolMode || 'AUTO'
        const lastFuncName = (/** @type {FunctionResponse[] | undefined}**/ opt.functionResponse)?.map(rsp => rsp.name)
        const mustSendNextTurn = ['searchImage', 'searchMusic', 'searchVideo']
        if (lastFuncName && lastFuncName?.find(name => mustSendNextTurn.includes(name))) {
          mode = 'ANY'
        }
        delete opt.toolMode
        body.tool_config = { function_calling_config: { mode } }
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
    }

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

    if (isProxy) {
      // --- 解析代理 (OpenAI 格式) 响应 ---
      const response = await result.json()
      if (this.debug) {
        console.log('Proxy Response:', JSON.stringify(response))
      }
      if (response.error) {
        logger.error(`[Gemini] 代理API返回错误: ${JSON.stringify(response.error)}`)
        throw new Error(JSON.stringify(response.error))
      }
      if (!response.choices || response.choices.length === 0) {
        logger.error(`[Gemini] 代理API无choices返回 - 完整响应: ${JSON.stringify(response)}`)
        // 无内容回复，可在此处添加重试逻辑
        throw new Error('Proxy API returned no choices.')
      }
      const message = response.choices[0].message
      responseContent = {
        role: 'model',
        parts: []
      }
      if (message.content) {
        responseContent.parts.push({ text: message.content })
      }
      if (message.tool_calls) {
        for (const toolCall of message.tool_calls) {
          try {
            responseContent.parts.push({
              functionCall: {
                name: toolCall.function.name,
                args: JSON.parse(toolCall.function.arguments)
              }
            })
          } catch (e) {
            logger.error(`Failed to parse tool call arguments from proxy: ${e}`)
          }
        }
      }
    } else {
      // --- 解析原生 Gemini 响应 ---
      /** @type {{candidates: Array<{content: Content, groundingMetadata: GroundingMetadata, finishReason: string}>}} */
      let response = await result.json()
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
      if (response.candidates[0].finishReason === 'MALFORMED_FUNCTION_CALL' && retryTime > 0) {
        logger.warn('Encountered MALFORMED_FUNCTION_CALL, retrying.')
        return this.sendMessage(text, opt, retryTime - 1)
      }
    }

    // --- 后续通用处理逻辑 ---
    if (responseContent.parts.filter(i => i.functionCall).length > 0) {
      // functionCall
      const functionCall = responseContent.parts.filter(i => i.functionCall).map(i => i.functionCall)
      const text = responseContent.parts.find(i => i.text)?.text
      if (text && text.trim()) {
        logger.info('send message: ' + text.trim())
        opt.replyPureTextCallback && await opt.replyPureTextCallback(text.trim())
      }
      let /** @type {FunctionResponse[]} **/ fcResults = []
      for (let fc of functionCall) {
        logger.info(`Executing function call: ${JSON.stringify(fc)}`)
        const funcName = fc.name
        let chosenTool = this.tools.find(t => t.name === funcName)
        let functionResponse = { name: funcName, response: { name: funcName, content: null } }
        if (!chosenTool) {
          functionResponse.response.content = { error: `Function ${funcName} doesn't exist` }
        } else {
          try {
            let isAdmin = ['admin', 'owner'].includes(this.e.sender.role) || (this.e.group?.is_admin && this.e.isMaster)
            let isOwner = ['owner'].includes(this.e.sender.role) || (this.e.group?.is_owner && this.e.isMaster)
            let args = Object.assign(fc.args, { isAdmin, isOwner, sender: this.e.sender.user_id, mode: 'gemini' })
            functionResponse.response.content = await chosenTool.func(args, this.e)
            if (this.debug) {
              logger.info(`Function result: ${JSON.stringify(functionResponse.response.content)}`)
            }
          } catch (err) {
            logger.error(err)
            functionResponse.response.content = { error: `Function execute error: ${err.message}` }
          }
        }
        fcResults.push(functionResponse)
      }
      let responseOpt = _.cloneDeep(opt)
      responseOpt.parentMessageId = idModel
      responseOpt.functionResponse = fcResults
      await this.upsertMessage(thisMessage)
      responseContent = handleSearchResponse(responseContent).responseContent
      const respMessage = Object.assign(responseContent, { id: idModel, parentMessageId: idThis })
      await this.upsertMessage(respMessage)
      return await this.sendMessage('', responseOpt)
    }
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
 * 将 Gemini 的 'parts' 数组转换为 OpenAI 的 'content' 格式。
 * 处理文本、多模态图像和函数响应。
 * @param {Array<object>} parts - 来自 Gemini 消息的 'parts' 数组。
 * @returns {string|Array<object>|null} - 用于 OpenAI 消息的 'content'。
 */
function convertGeminiPartsToOpenAIContent (parts) {
  if (!parts || parts.length === 0) {
    return null
  }

  // 首先检查函数响应，因为它们是一种特殊的消息类型
  const functionResponsePart = parts.find(p => p.functionResponse)
  if (functionResponsePart) {
    // 将函数响应表示为简单的文本字符串，供 LLM 理解。
    // 这避免了如果代理不能完美处理 'tool' 角色的复杂性。
    const funcResp = functionResponsePart.functionResponse
    return `Result for function call ${funcResp.name}: ${JSON.stringify(funcResp.response.content)}`
  }

  let textParts = []
  let imageParts = []

  // 处理文本和图像
  for (const part of parts) {
    if (part.text) {
      textParts.push(part.text)
    }
    if (part.inline_data && part.inline_data.data) {
      imageParts.push({
        type: 'image_url',
        image_url: {
          // 对于 OpenAI 格式，必须前缀 data URI scheme
          url: `data:${part.inline_data.mime_type};base64,${part.inline_data.data}`
        }
      })
    }
  }

  const combinedText = textParts.join('\n')

  if (imageParts.length > 0) {
    const contentArray = []
    if (combinedText) {
      contentArray.push({ type: 'text', text: combinedText })
    }
    contentArray.push(...imageParts)
    return contentArray
  } else {
    return combinedText || null
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
