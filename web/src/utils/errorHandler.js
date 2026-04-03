import { message } from 'ant-design-vue';

/**
 * 统一错误处理工具类
 */
export class ErrorHandler {
  static ERROR_CODE_MESSAGE_MAP = {
    AUTH_INVALID_KEY: '模型鉴权失败，请检查 API Key 与模型提供商配置。',
    MODEL_STREAM_FAILED: '模型服务调用失败，请稍后重试。',
    MODEL_STREAM_TIMEOUT: '模型响应超时，请重试或切换轻量模型。',
    MODEL_CIRCUIT_OPEN: '模型服务短时不可用，请稍后重试。',
    TOOL_EXECUTION_FAILED: '工具执行失败，已回退到基础回答。',
    STORAGE_WRITE_FAILED: '结果写入失败，请稍后重试。',
    CONTENT_GUARD_BLOCKED: '请求触发内容安全策略，已中断处理。',
    ANSWER_CONTRACT_HARD_BLOCK: '回答未满足最低证据与置信度要求，系统已转为澄清提问模式。',
  };
  /**
   * 处理通用错误
   * @param {Error} error - 错误对象
   * @param {string} context - 错误上下文
   * @param {Object} options - 配置选项
   */
  static handleError(error, context = '操作', options = {}) {
    const {
      showMessage = true,
      logToConsole = true,
      customMessage = null,
      severity = 'error'
    } = options;

    // 控制台日志
    if (logToConsole) {
      console.error(`${context}失败:`, error);
    }

    // 用户提示
    if (showMessage) {
      const displayMessage = customMessage || this.getErrorMessage(error, context);
      
      switch (severity) {
        case 'warning':
          message.warning(displayMessage);
          break;
        case 'info':
          message.info(displayMessage);
          break;
        case 'error':
        default:
          message.error(displayMessage);
          break;
      }
    }

    return error;
  }

  /**
   * 获取错误消息
   * @param {Error} error - 错误对象
   * @param {string} context - 错误上下文
   * @returns {string} 错误消息
   */
  static getErrorMessage(error, context) {
    const code = error?.error_code || error?.code;
    if (code && this.ERROR_CODE_MESSAGE_MAP[code]) {
      return this.ERROR_CODE_MESSAGE_MAP[code];
    }
    if (error?.message) {
      return `${context}失败: ${error.message}`;
    }
    return `${context}失败`;
  }

  /**
   * 处理网络请求错误
   * @param {Error} error - 错误对象
   * @param {string} context - 错误上下文
   */
  static handleNetworkError(error, context = '网络请求') {
    let customMessage = null;
    
    if (error?.code === 'NETWORK_ERROR') {
      customMessage = '网络连接失败，请检查网络设置';
    } else if (error?.status === 401) {
      customMessage = '认证失败，请重新登录';
    } else if (error?.status === 403) {
      customMessage = '权限不足，无法执行此操作';
    } else if (error?.status === 404) {
      customMessage = '请求的资源不存在';
    } else if (error?.status >= 500) {
      customMessage = '服务器错误，请稍后重试';
    }

    return this.handleError(error, context, { customMessage });
  }

  /**
   * 处理聊天相关错误
   * @param {Error} error - 错误对象
   * @param {string} operation - 操作类型
   */
  static handleChatError(error, operation) {
    const contextMap = {
      'send': '发送消息',
      'create': '创建对话',
      'delete': '删除对话',
      'rename': '重命名对话',
      'load': '加载对话',
      'export': '导出对话',
      'stream': '流式处理'
    };

    const context = contextMap[operation] || operation;
    return this.handleError(error, context);
  }

  /**
   * 处理验证错误
   * @param {string} message - 验证错误消息
   */
  static handleValidationError(message) {
    return this.handleError(
      new Error(message), 
      '输入验证', 
      { severity: 'warning', customMessage: message }
    );
  }

  /**
   * 处理异步操作错误
   * @param {Function} asyncFn - 异步函数
   * @param {string} context - 错误上下文
   * @param {Object} options - 配置选项
   */
  static async handleAsync(asyncFn, context, options = {}) {
    try {
      return await asyncFn();
    } catch (error) {
      this.handleError(error, context, options);
      throw error;
    }
  }

  /**
   * 创建错误处理装饰器
   * @param {string} context - 错误上下文
   * @param {Object} options - 配置选项
   */
  static createHandler(context, options = {}) {
    return (error) => this.handleError(error, context, options);
  }
}

/**
 * 快捷方法
 */
export const handleChatError = ErrorHandler.handleChatError.bind(ErrorHandler);
export const handleNetworkError = ErrorHandler.handleNetworkError.bind(ErrorHandler);
export const handleValidationError = ErrorHandler.handleValidationError.bind(ErrorHandler);
export const handleAsync = ErrorHandler.handleAsync.bind(ErrorHandler);

export default ErrorHandler;