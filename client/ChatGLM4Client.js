import { BaseClient } from './BaseClient.js'
import { newFetch } from '../utils/proxy.js'
import { createParser } from 'eventsource-parser'
import { getMessageById, upsertMessage } from '../utils/history.js'
import crypto from 'crypto'

export class ChatGLM4Client extends BaseClient {
  constructor (props) {
    if (!props.upsertMessage) {
      props.upsertMessage = async (message) => upsertMessage(message, 'ChatGLM4')
    }
    if (!props.getMessageById) {
      props.getMessageById = async (id) => getMessageById(id, 'ChatGLM4')
    }
    super(props)
    this.apiKey = props.apiKey
    this.model = props.model || 'glm-4'
    this.temperature = props.temperature ?? 0.8
    this.thinking = !!props.thinking
    this.baseUrl = props.baseUrl || 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
    this.debug = props.debug
  }

  async getHistory (parentMessageId) {
    const history = []
    let cursor = parentMessageId
    while (cursor) {
      const msg = await this.getMessageById(cursor)
      if (!msg) break
      history.push(msg)
      cursor = msg.parentMessageId
    }
    return history.reverse()
  }

  /**
   * @param text
   * @param {{parentMessageId: string?, stream: boolean?, onProgress: function?}} opt
   */
  async sendMessage (text, opt = {}) {
    const { parentMessageId, onProgress, system } = opt
    const history = await this.getHistory(parentMessageId)

    const messages = []
    if (system) {
      messages.push({ role: 'system', content: system })
    }

    messages.push(...history.map(m => ({
      role: m.role === 'User' ? 'user' : 'assistant',
      content: m.text || m.content
    })))

    messages.push({ role: 'user', content: text })

    const idThis = crypto.randomUUID()
    const thisMessage = {
      role: 'User',
      content: text,
      id: idThis,
      parentMessageId
    }
    await this.upsertMessage(thisMessage)

    const body = {
      model: this.model,
      messages,
      stream: true,
      temperature: this.temperature
    }

    // GLM-4.5, 4.7, 5, etc. support thinking
    const modelNum = parseFloat(this.model.replace(/[^\d.]/g, ''))
    if (modelNum >= 4.5 || this.model.includes('thinking')) {
      body.thinking = { type: this.thinking ? 'enabled' : 'disabled' }
    }

    let response
    try {
      response = await newFetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey}`
        },
        body: JSON.stringify(body)
      })
    } catch (fetchErr) {
      logger.error(`[ChatGLM] 网络请求失败 - URL: ${this.baseUrl}`)
      logger.error(`[ChatGLM] 错误详情: ${fetchErr.message}`)
      if (fetchErr.code) logger.error(`[ChatGLM] 错误码: ${fetchErr.code}`)
      if (fetchErr.cause) logger.error(`[ChatGLM] 错误原因: ${JSON.stringify(fetchErr.cause)}`)
      throw fetchErr
    }

    if (!response.ok) {
      const error = await response.text()
      logger.error(`[ChatGLM] API返回错误 - 状态码: ${response.status}, 响应体: ${error}`)
      throw new Error(`ChatGLM API error: ${response.status} ${error}`)
    }

    let fullText = ''
    let thinking_text = ''
    let messageId = crypto.randomUUID()

    return new Promise((resolve, reject) => {
      const parser = createParser((event) => {
        if (event.type === 'event') {
          if (event.data === '[DONE]') {
            const result = {
              text: fullText,
              id: messageId,
              parentMessageId: idThis,
              conversationId: '',
              thinking_text: thinking_text
            }
            this.upsertMessage({
              role: 'Assistant',
              text: fullText,
              thinking_text: thinking_text,
              id: messageId,
              parentMessageId: idThis
            }).then(() => resolve(result))
            return
          }

          try {
            const data = JSON.parse(event.data)
            const delta = data.choices?.[0]?.delta
            if (delta?.content) {
              fullText += delta.content
              if (onProgress) onProgress({ text: fullText })
            }
            if (delta?.reasoning_content) {
              thinking_text += delta.reasoning_content
            }
          } catch (e) {
            console.error('Error parsing ChatGLM SSE:', e)
          }
        }
      })

      response.body.on('data', (chunk) => {
        parser.feed(chunk.toString())
      })

      response.body.on('error', (err) => {
        logger.error(`[ChatGLM] SSE流错误: ${err.message || err}`)
        reject(err)
      })

      response.body.on('end', () => {
        // parser should have handled everything
      })
    })
  }
}
