import { Config, defaultOpenAIAPI } from '../utils/config.js'
import { getThinkingIntensity, reasoningEffortFor, resolveThinkingFormat } from '../utils/thinking.js'
import {
  extractContentFromFile,
  formatDate,
  getImg,
  getMasterQQ, getMaxModelTokens,
  getUin,
  getUserData,
  isCN
} from '../utils/common.js'
import { KeyvFile } from 'keyv-file'
import SydneyAIClient from '../utils/SydneyAIClient.js'
import { getChatHistoryGroup } from '../utils/chat.js'
import { OfficialChatGPTClient } from '../utils/message.js'
import { ClaudeAPIClient } from '../client/ClaudeAPIClient.js'
import { ClaudeAIClient } from '../utils/claude.ai/index.js'
import XinghuoClient from '../utils/xinghuo/xinghuo.js'
import { getMessageById, upsertMessage } from '../utils/history.js'
import { v4 as uuid } from 'uuid'
import fetch from 'node-fetch'
import { CustomGoogleGeminiClient } from '../client/CustomGoogleGeminiClient.js'
import { ChatGPTAPI } from '../utils/openai/chatgpt-api.js'
import { newFetch } from '../utils/proxy.js'
import { ChatGLM4Client } from '../client/ChatGLM4Client.js'
import { QwenApi } from '../utils/alibaba/qwen-api.js'
import { BingAIClient } from '../client/CopilotAIClient.js'
import Keyv from 'keyv'
import crypto from 'crypto'

export const roleMap = {
  owner: 'group owner',
  admin: 'group administrator'
}

const defaultPropmtPrefix = ', a large language model trained by OpenAI. You answer as concisely as possible for each response (e.g. don’t be verbose). It is very important that you answer as concisely as possible, so please remember this. If you are generating a list, do not have too many items. Keep the number of items short.'

async function handleSystem (e, system, settings) {
  if (settings.enableGroupContext) {
    try {
      let opt = {}
      opt.groupId = e.group_id
      opt.qq = e.sender.user_id
      opt.nickname = e.sender.card
      opt.groupName = e.group.name || e.group_name
      opt.botName = e.isGroup ? (e.group.pickMember(getUin(e)).card || e.group.pickMember(getUin(e)).nickname) : e.bot.nickname
      let master = (await getMasterQQ())[0]
      if (master && e.group) {
        opt.masterName = e.group.pickMember(parseInt(master)).card || e.group.pickMember(parseInt(master)).nickname
      }
      if (master && !e.group) {
        opt.masterName = e.bot.getFriendList().get(parseInt(master))?.nickname
      }
      let chats = await getChatHistoryGroup(e, Config.groupContextLength)
      opt.chats = chats
      const namePlaceholder = '[name]'
      const defaultBotName = 'ChatGPT'
      const groupContextTip = Config.groupContextTip
      system = system.replaceAll(namePlaceholder, opt.botName || defaultBotName) +
        ((opt.groupId) ? groupContextTip : '')
      system += 'Attention, you are currently chatting in a qq group, then one who asks you now is' + `${opt.nickname}(${opt.qq})。`
      system += `the group name is ${opt.groupName}, group id is ${opt.groupId}。`
      if (opt.botName) {
        system += `Your nickname is ${opt.botName} in the group,`
      }
      if (chats) {
        system += 'There is the conversation history in the group, you must chat according to the conversation history context"'
        system += chats
          .map(chat => {
            let sender = chat.sender || {}
            // if (sender.user_id === e.bot.uin && chat.raw_message.startsWith('建议的回复')) {
            if (chat.raw_message.startsWith('建议的回复')) {
              // 建议的回复太容易污染设定导致对话太固定跑偏了
              return ''
            }
            return `【${sender.card || sender.nickname}】(qq：${sender.user_id}, ${roleMap[sender.role] || 'normal user'}，${sender.area ? 'from ' + sender.area + ', ' : ''} ${sender.age} years old, 群头衔：${sender.title}, gender: ${sender.sex}, time：${formatDate(new Date(chat.time * 1000))}, messageId: ${chat.message_id}) 说：${chat.raw_message}`
          })
          .join('\n')
      }
    } catch (err) {
      if (e.isGroup) {
        logger.warn('获取群聊聊天记录失败，本次对话不携带聊天记录', err)
      }
    }
  }
  return system
}

async function getImagePayloads (e) {
  const imageUrls = await getImg(e)
  if (!imageUrls?.length) {
    return []
  }

  const images = []
  for (const imageUrl of imageUrls) {
    const response = await fetch(imageUrl)
    if (!response.ok) {
      throw new Error(`Failed to fetch image ${imageUrl}: ${response.status}`)
    }
    const mimeType = response.headers.get('content-type')?.split(';')[0] || 'image/jpeg'
    images.push({
      data: Buffer.from(await response.arrayBuffer()).toString('base64'),
      mimeType
    })
  }
  return images
}

class Core {
  async sendMessage (prompt, conversation = {}, use, e, opt = {
    system: {
      api: Config.promptPrefixOverride,
      qwen: Config.promptPrefixOverride,
      bing: Config.sydney,
      claude: Config.claudeSystemPrompt,
      claude2: Config.claudeSystemPrompt,
      gemini: Config.geminiPrompt,
      xh: Config.xhPrompt,
      chatglm: Config.chatglmPrompt
    },
    settings: {
      replyPureTextCallback: undefined,
      enableGroupContext: Config.enableGroupContext
    }
  }) {
    if (!conversation) {
      conversation = {
        timeoutMs: Config.defaultTimeoutMs
      }
    }
    if (Config.debug) {
      logger.mark(`using ${use} mode`)
    }
    const userData = await getUserData(e.user_id)
    const useCast = userData.cast || {}
    if (use === 'bing') {
      const cacheOptions = {
        namespace: Config.toneStyle,
        store: new KeyvFile({ filename: 'cache.json' })
      }
      const conversationsCache = new Keyv(cacheOptions)
      let client = new BingAIClient(Config.bingAiToken, Config.sydneyReverseProxy, Config.debug, Config._2captchaKey, Config.bingAiClientId, Config.bingAiScope, Config.bingAiRefreshToken, Config.bingAiOid, Config.bingReasoning)
      const conversationKey = `SydneyUser_${e.sender.user_id}`
      const conversations = (await conversationsCache.get(conversationKey)) || {
        messages: [],
        createdAt: Date.now()
      }
      if (Config.debug) {
        logger.debug(JSON.stringify(conversations))
      }
      const previousCachedMessages = SydneyAIClient.getMessagesForConversation(conversations.messages, conversation.parentMessageId)
        .map((message) => {
          return {
            text: message.message,
            author: message.role === 'User' ? 'user' : 'bot'
          }
        })
      let system = opt.system.bing
      if (opt.settings.enableGroupContext && e.isGroup) {
        let chats = await getChatHistoryGroup(e, Config.groupContextLength)
        const namePlaceholder = '[name]'
        const defaultBotName = 'Copilot'
        const groupContextTip = Config.groupContextTip
        let botName = e.isGroup ? (e.group.pickMember(getUin(e)).card || e.group.pickMember(getUin(e)).nickname) : e.bot.nickname
        system = system.replaceAll(namePlaceholder, botName || defaultBotName) +
          ((opt.settings.enableGroupContext && e.group_id) ? groupContextTip : '')
        system += 'Attention, you are currently chatting in a qq group, then one who asks you now is' + `${e.sender.card || e.sender.nickname}(${e.sender.user_id}).`
        system += `the group name is ${e.group.name || e.group_name}, group id is ${e.group_id}.`
        system += `Your nickname is ${botName} in the group,`
        if (chats) {
          system += 'There is the conversation history in the group, you must chat according to the conversation history context"'
          system += chats
            .map(chat => {
              let sender = chat.sender || {}
              return `【${sender.card || sender.nickname}】(qq：${sender.user_id}, ${roleMap[sender.role] || 'normal user'}，${sender.area ? 'from ' + sender.area + ', ' : ''} ${sender.age} years old, 群头衔：${sender.title}, gender: ${sender.sex}, time：${formatDate(new Date(chat.time * 1000))}, messageId: ${chat.message_id}) 说：${chat.raw_message}`
            })
            .join('\n')
        }
      }
      const msg = `System:\n${system}\n\nPrevious Messages:\n${JSON.stringify(previousCachedMessages)}\n\nUser: ${prompt}`
      const response = await client.sendMessage(msg)
      logger.info({ response })
      const userMessage = {
        id: crypto.randomUUID(),
        parentMessageId: conversation.parentMessageId,
        role: 'User',
        message: prompt
      }
      conversations.messages.push(userMessage)
      const replyMessage = {
        id: crypto.randomUUID(),
        parentMessageId: userMessage.id,
        role: 'Bing',
        message: response
      }
      conversations.messages.push(replyMessage)
      await conversationsCache.set(conversationKey, conversations)
      return {
        text: response,
        parentMessageId: replyMessage.id

      }
    } else if (use === 'api3') {
      // official without cloudflare
      let accessToken = await redis.get('CHATGPT:TOKEN')
      // if (!accessToken) {
      //   throw new Error('未绑定ChatGPT AccessToken，请使用#chatgpt设置token命令绑定token')
      // }
      this.chatGPTApi = new OfficialChatGPTClient({
        accessToken,
        apiReverseUrl: Config.api,
        timeoutMs: 120000
      })
      let sendMessageResult = await this.chatGPTApi.sendMessage(prompt, conversation)
      // 更新最后一条prompt
      await redis.set(`CHATGPT:CONVERSATION_LAST_MESSAGE_PROMPT:${sendMessageResult.conversationId}`, prompt)
      // 更新最后一条messageId
      await redis.set(`CHATGPT:CONVERSATION_LAST_MESSAGE_ID:${sendMessageResult.conversationId}`, sendMessageResult.id)
      await redis.set(`CHATGPT:QQ_CONVERSATION:${(e.isGroup && Config.groupMerge) ? e.group_id.toString() : e.sender.user_id}`, sendMessageResult.conversationId)
      if (!conversation.conversationId) {
        // 如果是对话的创建者
        await redis.set(`CHATGPT:CONVERSATION_CREATER_ID:${sendMessageResult.conversationId}`, e.sender.user_id)
        await redis.set(`CHATGPT:CONVERSATION_CREATER_NICK_NAME:${sendMessageResult.conversationId}`, e.sender.card)
      }
      (async () => {
        let audio = await this.chatGPTApi.synthesis(sendMessageResult)
        if (audio) {
          await e.reply(segment.record(audio))
        }
      })().catch(err => {
        logger.warn('发送语音失败', err)
      })
      return sendMessageResult
    } else if (use === 'claude') {
      // slack已经不可用，移除
      let keys = Config.claudeApiKey?.split(/[,;]/).map(key => key.trim()).filter(key => key)
      let choiceIndex = Math.floor(Math.random() * keys.length)
      let key = keys[choiceIndex]
      logger.info(`使用API Key：${key}`)
      while (keys.length >= 0) {
        let errorMessage = ''
        const client = new ClaudeAPIClient({
          key,
          model: Config.claudeApiModel || 'claude-3-sonnet-20240229',
          debug: true,
          baseUrl: Config.claudeApiBaseUrl
          // temperature: Config.claudeApiTemperature || 0.5
        })
        let option = {
          stream: false,
          parentMessageId: conversation.parentMessageId,
          conversationId: conversation.conversationId,
          system: opt.system.claude,
          max_tokens: Config.apiMaxToken
        }
        const thinkingIntensity = getThinkingIntensity()
        if (thinkingIntensity !== 'default' && thinkingIntensity !== 'off') {
          // claude不传thinking即为关闭，off无需处理
          option.thinking = { type: 'enabled', budget_tokens: { low: 1024, medium: 4096, high: 16384 }[thinkingIntensity] }
        }
        if (opt.settings.enableGroupContext && e.isGroup) {
          let chats = await getChatHistoryGroup(e, Config.groupContextLength)
          const namePlaceholder = '[name]'
          const defaultBotName = 'GeminiPro'
          const groupContextTip = Config.groupContextTip
          let botName = e.isGroup ? (e.group.pickMember(getUin(e)).card || e.group.pickMember(getUin(e)).nickname) : e.bot.nickname
          option.system = option.system.replaceAll(namePlaceholder, botName || defaultBotName) +
            ((opt.settings.enableGroupContext && e.group_id) ? groupContextTip : '')
          option.system += 'Attention, you are currently chatting in a qq group, then one who asks you now is' + `${e.sender.card || e.sender.nickname}(${e.sender.user_id}).`
          option.system += `the group name is ${e.group.name || e.group_name}, group id is ${e.group_id}.`
          option.system += `Your nickname is ${botName} in the group,`
          if (chats) {
            option.system += 'There is the conversation history in the group, you must chat according to the conversation history context"'
            option.system += chats
              .map(chat => {
                let sender = chat.sender || {}
                return `【${sender.card || sender.nickname}】(qq：${sender.user_id}, ${roleMap[sender.role] || 'normal user'}，${sender.area ? 'from ' + sender.area + ', ' : ''} ${sender.age} years old, 群头衔：${sender.title}, gender: ${sender.sex}, time：${formatDate(new Date(chat.time * 1000))}, messageId: ${chat.message_id}) 说：${chat.raw_message}`
              })
              .join('\n')
          }
        }
        let img = await getImg(e)
        if (img && img.length > 0) {
          const response = await fetch(img[0])
          const base64Image = Buffer.from(await response.arrayBuffer()).toString('base64')
          opt.image = base64Image
        }
        try {
          let rsp = await client.sendMessage(prompt, option)
          return rsp
        } catch (err) {
          errorMessage = err.message
          switch (err.message) {
            case 'rate_limit_error': {
              // api没钱了或者当月/日/时/分额度耗尽
              // throw new Error('claude API额度耗尽或触发速率限制')
              break
            }
            case 'authentication_error': {
              // 无效的key
              // throw new Error('claude API key无效')
              break
            }
            default:
          }
          logger.error(`[Claude] API错误 - Key: ${key?.slice(0, 8)}***, 错误信息: ${errorMessage}`)
          if (err.code) logger.error(`[Claude] 错误码: ${err.code}`)
          if (err.stack) logger.error(`[Claude] 错误堆栈: ${err.stack}`)
          logger.warn(`claude api 错误：[${key}] ${errorMessage}`)
        }
        if (keys.length === 0) {
          throw new Error(errorMessage)
        }
        keys.splice(choiceIndex, 1)
        choiceIndex = Math.floor(Math.random() * keys.length)
        key = keys[choiceIndex]
        logger.info(`使用API Key：${key}`)
      }
    } else if (use === 'claude2') {
      let { conversationId } = conversation
      let client = new ClaudeAIClient({
        organizationId: Config.claudeAIOrganizationId,
        sessionKey: Config.claudeAISessionKey,
        debug: Config.debug,
        proxy: Config.proxy
      })
      let toSummaryFileContent
      try {
        if (e.source) {
          let msgs = e.isGroup ? await e.group.getChatHistory(e.source.seq, 1) : await e.friend.getChatHistory(e.source.time, 1)
          let sourceMsg = msgs[0]
          let fileMsgElem = sourceMsg.message.find(msg => msg.type === 'file')
          if (fileMsgElem) {
            toSummaryFileContent = await extractContentFromFile(fileMsgElem, e)
          }
        }
      } catch (err) {
        logger.warn('读取文件内容出错， 忽略文件内容', err)
      }

      let attachments = []
      if (toSummaryFileContent?.content) {
        attachments.push({
          extracted_content: toSummaryFileContent.content,
          file_name: toSummaryFileContent.name,
          file_type: 'pdf',
          file_size: 200312,
          totalPages: 20
        })
        logger.info(toSummaryFileContent.content)
      }
      if (conversationId) {
        return await client.sendMessage(prompt, conversationId, attachments)
      } else {
        let conv = await client.createConversation()
        return await client.sendMessage(prompt, conv.uuid, attachments)
      }
    } else if (use === 'xh') {
      const cacheOptions = {
        namespace: 'xh',
        store: new KeyvFile({ filename: 'cache.json' })
      }
      const ssoSessionId = Config.xinghuoToken
      if (!ssoSessionId) {
        // throw new Error('未绑定星火token，请使用#chatgpt设置星火token命令绑定token。（获取对话页面的ssoSessionId cookie值）')
        logger.warn('未绑定星火token，请使用#chatgpt设置星火token命令绑定token。（获取对话页面的ssoSessionId cookie值）')
      }
      let client = new XinghuoClient({
        ssoSessionId,
        cache: cacheOptions
      })
      // 获取图片资源
      const image = await getImg(e)
      try {
        let response = await client.sendMessage(prompt, {
          e,
          chatId: conversation?.conversationId,
          image: image ? image[0] : undefined,
          system: opt.system.xh
        })
        return response
      } catch (err) {
        logger.error(`[星火] sendMessage 错误: ${err.message || err}`)
        if (err.code) logger.error(`[星火] 错误码: ${err.code}`)
        if (err.stack) logger.error(`[星火] 错误堆栈: ${err.stack}`)
        throw err
      }
    } else if (use === 'azure') {
      let azureModel
      try {
        azureModel = await import('@azure/openai')
      } catch (error) {
        throw new Error('未安装@azure/openai包，请执行pnpm install @azure/openai安装')
      }
      let OpenAIClient = azureModel.OpenAIClient
      let AzureKeyCredential = azureModel.AzureKeyCredential
      let msg = conversation.messages
      let content = {
        role: 'user',
        content: prompt
      }
      msg.push(content)
      const client = new OpenAIClient(Config.azureUrl, new AzureKeyCredential(Config.azApiKey))
      const deploymentName = Config.azureDeploymentName
      const { choices } = await client.getChatCompletions(deploymentName, msg)
      let completion = choices[0].message
      return {
        text: completion.content,
        message: completion
      }
    } else if (use === 'qwen') {
      let completionParams = {
        parameters: {
          top_p: Config.qwenTopP || 0.5,
          top_k: Config.qwenTopK || 50,
          seed: Config.qwenSeed > 0 ? Config.qwenSeed : Math.floor(Math.random() * 114514),
          temperature: Config.qwenTemperature || 1,
          enable_search: !!Config.qwenEnableSearch,
          result_format: 'message'
        }
      }
      const thinkingIntensity = getThinkingIntensity()
      if (thinkingIntensity !== 'default') {
        completionParams.parameters.enable_thinking = thinkingIntensity !== 'off'
        if (thinkingIntensity !== 'off') {
          completionParams.parameters.thinking_budget = { low: 1024, medium: 8192, high: 32768 }[thinkingIntensity]
        }
      }
      if (Config.qwenModel) {
        completionParams.model = Config.qwenModel
      }
      const currentDate = new Date().toISOString().split('T')[0]

      async function um (message) {
        return await upsertMessage(message, 'QWEN')
      }

      async function gm (id) {
        return await getMessageById(id, 'QWEN')
      }

      let opts = {
        apiKey: Config.qwenApiKey,
        debug: Config.debug,
        upsertMessage: um,
        getMessageById: gm,
        systemMessage: `You are ${Config.assistantLabel} ${useCast?.api || opt.system.qwen || defaultPropmtPrefix}
        Current date: ${currentDate}`,
        completionParams,
        assistantLabel: Config.assistantLabel,
        fetch: newFetch
      }

      let option = {
        timeoutMs: 600000,
        completionParams
      }
      if (conversation) {
        if (!conversation.conversationId) {
          conversation.conversationId = uuid()
        }
        option = Object.assign(option, conversation)
      }
      const images = await getImagePayloads(e)
      if (images.length > 0) {
        option.images = images
      }
      let msg
      try {
        this.qwenApi = new QwenApi(opts)
        msg = await this.qwenApi.sendMessage(prompt, option)
      } catch (err) {
        logger.error(`[Qwen] sendMessage错误: ${err.message || err}`)
        if (err.code) logger.error(`[Qwen] 错误码: ${err.code}`)
        if (err.statusCode) logger.error(`[Qwen] HTTP状态码: ${err.statusCode}`)
        if (err.stack) logger.error(`[Qwen] 错误堆栈: ${err.stack}`)
        throw err
      }
      return msg
    } else if (use === 'gemini') {
      let client = new CustomGoogleGeminiClient({
        e,
        userId: e.sender.user_id,
        key: Config.getGeminiKey(),
        model: Config.geminiModel,
        baseUrl: Config.geminiBaseUrl,
        debug: Config.debug
      })
      let option = {
        stream: false,
        onProgress: (data) => {
          if (Config.debug) {
            logger.info(data)
          }
        },
        parentMessageId: conversation.parentMessageId,
        conversationId: conversation.conversationId,
        search: Config.geminiEnableGoogleSearch,
        codeExecution: Config.geminiEnableCodeExecution
      }
      const image = await getImg(e)
      let imageUrl = image ? image[0] : undefined
      if (imageUrl) {
        const response = await fetch(imageUrl)
        const base64Image = Buffer.from(await response.arrayBuffer())
        option.image = base64Image.toString('base64')
      }
      let system = opt.system.gemini
      if (opt.settings.enableGroupContext && e.isGroup) {
        let chats = await getChatHistoryGroup(e, Config.groupContextLength)
        const namePlaceholder = '[name]'
        const defaultBotName = 'GeminiPro'
        const groupContextTip = Config.groupContextTip
        let botName = e.isGroup ? (e.group.pickMember(getUin(e)).card || e.group.pickMember(getUin(e)).nickname) : e.bot.nickname
        system = system.replaceAll(namePlaceholder, botName || defaultBotName) +
          ((opt.settings.enableGroupContext && e.group_id) ? groupContextTip : '')
        system += 'Attention, you are currently chatting in a qq group, then one who asks you now is' + `${e.sender.card || e.sender.nickname}(${e.sender.user_id}).`
        system += `the group name is ${e.group.name || e.group_name}, group id is ${e.group_id}.`
        system += `Your nickname is ${botName} in the group,`
        if (chats) {
          system += 'There is the conversation history in the group, you must chat according to the conversation history context"'
          system += chats
            .map(chat => {
              let sender = chat.sender || {}
              return `【${sender.card || sender.nickname}】(qq：${sender.user_id}, ${roleMap[sender.role] || 'normal user'}，${sender.area ? 'from ' + sender.area + ', ' : ''} ${sender.age} years old, 群头衔：${sender.title}, gender: ${sender.sex}, time：${formatDate(new Date(chat.time * 1000))}, messageId: ${chat.message_id}) 说：${chat.raw_message}`
            })
            .join('\n')
        }
      }
      if (Config.enableChatSuno) {
        system += 'If I ask you to generate music or write songs, you need to reply with information suitable for Suno to generate music. Please use keywords such as Verse, Chorus, Bridge, Outro, and End to segment the lyrics, such as [Verse 1], The returned message is in JSON format, with a structure of ```json{"option": "Suno", "tags": "style", "title": "title of the song", "lyrics": "lyrics"}```.'
      }
      option.system = system
      option.replyPureTextCallback = opt.settings.replyPureTextCallback || (async (msg) => {
        if (msg) {
          await e.reply(msg, true)
        }
      })
      try {
        return await client.sendMessage(prompt, option)
      } catch (err) {
        logger.error(`[Gemini] sendMessage 错误: ${err.message || err}`)
        if (err.code) logger.error(`[Gemini] 错误码: ${err.code}`)
        if (err.stack) logger.error(`[Gemini] 错误堆栈: ${err.stack}`)
        throw err
      }
    } else if (use === 'chatglm4') {
      const thinkingIntensity = getThinkingIntensity()
      const client = new ChatGLM4Client({
        apiKey: Config.chatglmApiKey,
        model: Config.chatglmModel,
        thinking: thinkingIntensity !== 'default' ? thinkingIntensity !== 'off' : Config.chatglmThinking,
        temperature: Config.chatglmTemperature,
        debug: Config.debug
      })
      let system = await handleSystem(e, opt.system.chatglm, opt.settings)
      try {
        let resp = await client.sendMessage(prompt, { ...conversation, system })
        if (resp.image) {
          this.reply(segment.image(resp.image), true)
        }
        return resp
      } catch (err) {
        logger.error(`[ChatGLM4] sendMessage 错误: ${err.message || err}`)
        if (err.code) logger.error(`[ChatGLM4] 错误码: ${err.code}`)
        if (err.stack) logger.error(`[ChatGLM4] 错误堆栈: ${err.stack}`)
        throw err
      }
    } else {
      // openai api
      let completionParams = {}
      if (Config.model) {
        completionParams.model = Config.model
      }
      const thinkingIntensity = getThinkingIntensity()
      if (thinkingIntensity !== 'default') {
        completionParams.reasoning_effort = reasoningEffortFor(thinkingIntensity, resolveThinkingFormat(Config.openAiBaseUrl))
      }
      const currentDate = new Date().toISOString().split('T')[0]
      let promptPrefix = `You are ${Config.assistantLabel} ${useCast?.api || opt.system.api || defaultPropmtPrefix}
        Current date: ${currentDate}`
      let maxModelTokens = getMaxModelTokens(completionParams.model)
      // let system = promptPrefix
      let system = await handleSystem(e, promptPrefix, opt.settings)
      if (Config.enableChatSuno) {
        system += 'If I ask you to generate music or write songs, you need to reply with information suitable for Suno to generate music. Please use keywords such as Verse, Chorus, Bridge, Outro, and End to segment the lyrics, such as [Verse 1], The returned song information needs to be wrapped in JSON format and sent to me in Markdown format. The message structure is ` ` JSON {"option": "Suno", "tags": "style", "title": "title of The Song", "lyrics": "lyrics"} `.'
      }
      logger.debug(system)
      let opts = {
        apiBaseUrl: Config.openAiBaseUrl,
        apiKey: Config.apiKey,
        debug: false,
        upsertMessage,
        getMessageById,
        systemMessage: system,
        completionParams,
        assistantLabel: Config.assistantLabel,
        fetch: newFetch,
        maxModelTokens,
        maxResponseTokens: Config.apiMaxToken
      }
      let openAIAccessible = (Config.proxy || !(await isCN())) // 配了代理或者服务器在国外，默认认为不需要反代
      if (opts.apiBaseUrl !== defaultOpenAIAPI && openAIAccessible && !Config.openAiForceUseReverse) {
        // 如果配了proxy(或者不在国内)，而且有反代，但是没开启强制反代,将baseurl删掉
        delete opts.apiBaseUrl
      }
      // const client = new OpenAI({
      //   apiKey: Config.apiKey,
      //   baseURL: opts.apiBaseUrl,
      //   fetch: newFetch
      // })

      this.chatGPTApi = new ChatGPTAPI(opts)
      let option = {
        timeoutMs: 600000,
        completionParams,
        stream: Config.apiStream,
        onProgress: (data) => {
          if (Config.debug) {
            logger.info(data?.text || data.functionCall || data)
          }
        }
        // systemMessage: promptPrefix
      }
      option.systemMessage = system
      if (conversation) {
        if (!conversation.conversationId) {
          conversation.conversationId = uuid()
        }
        option = Object.assign(option, conversation)
      }
      const images = await getImagePayloads(e)
      if (images.length > 0) {
        option.images = images
      }
      let msg
      try {
        msg = await this.chatGPTApi.sendMessage(prompt, option)
      } catch (err) {
        if (err.message?.indexOf('context_length_exceeded') > 0) {
          logger.warn(err)
          await redis.del(`CHATGPT:CONVERSATIONS:${e.sender.user_id}`)
          await redis.del(`CHATGPT:WRONG_EMOTION:${e.sender.user_id}`)
          await this.reply('字数超限啦，将为您自动结束本次对话。')
          return null
        } else {
          logger.error(`[OpenAI] sendMessage错误: ${err.message || err}`)
          if (err.code) logger.error(`[OpenAI] 错误码: ${err.code}`)
          if (err.statusCode) logger.error(`[OpenAI] HTTP状态码: ${err.statusCode}`)
          if (err.stack) logger.error(`[OpenAI] 错误堆栈: ${err.stack}`)
          throw err
        }
      }
      return msg
    }
  }
}

export default new Core()
