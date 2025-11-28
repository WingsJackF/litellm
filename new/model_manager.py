"""
简化版模型管理器
基于 LiteLLM 的设计思路，实现模型的注册、识别和管理功能
支持统一的 API 调用接口 (model, messages, tools, response_format)
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


# 加载 .env 文件
load_dotenv()


@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str
    provider: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    default_params: Dict = field(default_factory=dict)
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    max_tokens: Optional[int] = None
    
    # 新增字段，用于指示是否使用特殊的 responses API (如 gpt-5)
    use_responses_api: bool = False
    output_version: Optional[str] = None


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Token 使用统计回调处理程序"""
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = None
        self.end_time = None
        self.total_duration = 0.0
        self.call_count = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts running."""
        self.start_time = time.time()

    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends running."""
        if self.start_time is not None:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_duration += duration
            self.call_count += 1
        
        usage = None
        
        # Handle LLMResult
        if hasattr(response, "llm_output") and response.llm_output:
            if "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
        
        # Handle direct usage_metadata
        elif hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            
        if usage:
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
            total_tokens = usage.get("total_tokens", 0)
            
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += total_tokens
            
            # print(f"| Model: {self.model_name} | Tokens: {input_tokens} in, {output_tokens} out | Time: {duration:.2f}s")


class ModelManager:
    """
    模型管理器 - 管理和识别不同的 LLM 模型
    
    功能：
    1. 注册和管理模型配置
    2. 统一的 API 调用接口 (chat)
    3. 自动实例化对应的 LangChain 客户端
    """
    
    def __init__(self, model_file: str = "model.json"):
        """
        初始化模型管理器
        
        Args:
            model_file: 模型配置文件路径，默认为 model.json
        """
        self.model_file = Path(__file__).parent / model_file
        self.models: Dict[str, ModelConfig] = {}
        self.model_aliases: Dict[str, str] = {}
        
        # 缓存已实例化的模型客户端
        self._client_cache: Dict[str, Any] = {}
        
        # 提供商列表
        self.providers: List[str] = [
            "openai", "anthropic", "google", "deepseek", "ollama"
        ]
        
        # 提供商默认 API Base URL 映射
        self.provider_api_bases: Dict[str, str] = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
            "google": "https://generativelanguage.googleapis.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "ollama": "http://localhost:11434/v1",
        }
        
        # 已知模型配置
        self.known_models: Dict[str, str] = {
            "gpt-4o": "openai",
            "gpt-4-turbo": "openai",
            "claude-3-5-sonnet-20241022": "anthropic",
            "gemini-1.5-pro": "google",
            "deepseek-chat": "deepseek"
        }
        
        self._initialize_default_models()
        self._load_from_json()
    
    def _initialize_default_models(self):
        """初始化默认支持的模型配置"""
        for model_name, provider in self.known_models.items():
            config = ModelConfig(
                model_name=model_name,
                provider=provider,
                api_base=self.provider_api_bases.get(provider),
                supports_vision="gpt-4" in model_name or "claude" in model_name or "gemini" in model_name
            )
            self.models[model_name] = config
    
    def _load_from_json(self):
        """从 JSON 文件加载自定义模型配置"""
        try:
            if self.model_file.exists():
                with open(self.model_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'custom_models' in data:
                    for model_data in data['custom_models']:
                        config = ModelConfig(
                            model_name=model_data['model_name'],
                            provider=model_data['provider'],
                            api_base=model_data.get('api_base'),
                            api_key=model_data.get('api_key'),
                            default_params=model_data.get('default_params', {}),
                            supports_streaming=model_data.get('supports_streaming', True),
                            supports_functions=model_data.get('supports_functions', False),
                            supports_vision=model_data.get('supports_vision', False),
                            max_tokens=model_data.get('max_tokens'),
                            use_responses_api=model_data.get('use_responses_api', False),
                            output_version=model_data.get('output_version')
                        )
                        self.models[model_data['model_name']] = config
                
                if 'aliases' in data:
                    self.model_aliases = data['aliases']
                    
        except Exception as e:
            print(f"⚠️  加载模型配置失败: {e}")
    
    def _save_to_json(self):
        """保存自定义模型配置到 JSON 文件"""
        try:
            custom_models = []
            for model_name, config in self.models.items():
                if model_name not in self.known_models:
                    model_data = asdict(config)
                    custom_models.append(model_data)
            
            data = {
                'custom_models': custom_models,
                'aliases': self.model_aliases
            }
            
            with open(self.model_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存模型配置失败: {e}")

    def register_model(self, model_name: str, provider: str, **kwargs):
        """注册新模型"""
        # 提取 api_base，避免重复传递
        api_base = kwargs.pop('api_base', None) or self.provider_api_bases.get(provider)
        
        config = ModelConfig(
            model_name=model_name,
            provider=provider,
            api_base=api_base,
            **kwargs
        )
        self.models[model_name] = config
        self._save_to_json()
        # 清除缓存
        if model_name in self._client_cache:
            del self._client_cache[model_name]
        return config

    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        if model in self.model_aliases:
            model = self.model_aliases[model]
        return self.models.get(model)

    def _get_api_key(self, provider: str, config_key: Optional[str] = None) -> Optional[str]:
        """获取 API Key (优先使用统一的 API_KEY)"""
        # 1. 优先检查环境变量中的统一 API_KEY
        unified_key = os.getenv("API_KEY")
        if unified_key:
            return unified_key
            
        # 2. 如果配置中有显式的 key，使用它
        if config_key:
            return config_key
        
        # 3. 检查提供商特定的环境变量
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        return os.getenv(env_keys.get(provider, ""))

    def _get_api_base(self, provider: str, config_base: Optional[str] = None) -> Optional[str]:
        """获取 API Base (优先使用统一的 BASE_URL)"""
        # 1. 优先检查环境变量中的统一 BASE_URL
        unified_base = os.getenv("BASE_URL")
        if unified_base:
            return unified_base
            
        # 2. 检查提供商特定的环境变量
        env_var = f"{provider.upper()}_API_BASE"
        env_base = os.getenv(env_var)
        if env_base:
            return env_base

        # 3. 使用配置中的 base (或者默认值)
        return config_base

    def _create_client(self, config: ModelConfig):
        """创建 LangChain 客户端实例"""
        api_key = self._get_api_key(config.provider, config.api_key)
        api_base = self._get_api_base(config.provider, config.api_base)
        
        if not api_key and config.provider != "ollama":
             print(f"⚠️  Warning: No API key found for {config.provider}")

        callbacks = [TokenUsageCallbackHandler(config.model_name)]
        
        common_args = {
            "model": config.model_name,
            "api_key": api_key,
            "base_url": api_base,
            "callbacks": callbacks,
            "max_tokens": config.max_tokens,
            **config.default_params
        }
        
        # 移除 None 值参数
        common_args = {k: v for k, v in common_args.items() if v is not None}

        if config.provider == "openai" or config.provider == "deepseek":
            # DeepSeek 兼容 OpenAI 接口
            if config.use_responses_api:
                common_args["use_responses_api"] = True
                if config.output_version:
                    common_args["output_version"] = config.output_version
            return ChatOpenAI(**common_args)
            
        elif config.provider == "anthropic":
            return ChatAnthropic(**common_args)
            
        elif config.provider == "google":
            # Google GenAI 参数稍有不同
            if "base_url" in common_args:
                del common_args["base_url"] # Google usually doesn't use base_url this way in LangChain
            return ChatGoogleGenerativeAI(**common_args)
            
        elif config.provider == "ollama":
            # Ollama use ChatOpenAI compatible endpoint usually
            return ChatOpenAI(**common_args)
            
        else:
            # 默认尝试用 ChatOpenAI 兼容模式
            return ChatOpenAI(**common_args)

    def get_model(self, model_name: str):
        """获取模型实例 (带缓存)"""
        if model_name in self.model_aliases:
            model_name = self.model_aliases[model_name]
            
        if model_name in self._client_cache:
            return self._client_cache[model_name]
            
        config = self.get_model_config(model_name)
        if not config:
            # 尝试作为 OpenAI 兼容模型直接创建
            config = ModelConfig(model_name=model_name, provider="openai")
            
        client = self._create_client(config)
        self._client_cache[model_name] = client
        return client

    def chat(
        self, 
        model: str, 
        messages: List[Any], 
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
        use_responses_api: Optional[bool] = None,
        **kwargs
    ) -> Dict:
        """
        统一 API 调用接口 - 返回原始 JSON 格式响应
        
        Args:
            model: 模型名称
            messages: 已格式化的消息列表 (LangChain Message 对象)
            tools: 工具定义列表
            response_format: 响应格式定义
            stream: 是否流式输出
            use_responses_api: 是否使用 responses API（None 时使用配置文件设置）
            **kwargs: 其他参数
        
        Returns:
            Dict: 原始 OpenAI API 格式的 JSON 响应
            格式: {
                "id": "chatcmpl-xxx",
                "model": "gpt-4o",
                "choices": [{"message": {"content": "...", "role": "assistant"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
        """
        # 获取模型配置
        config = self.get_model_config(model) or ModelConfig(model, "openai")
        api_key = self._get_api_key(config.provider, config.api_key)
        api_base = self._get_api_base(config.provider, config.api_base)
        
        # 确定使用哪种 API：优先使用参数，其次使用配置
        if use_responses_api is None:
            use_responses_api = config.use_responses_api
        
        # 转换 LangChain Messages 为 API 格式
        from message_manager import MessageManager
        msg_manager = MessageManager(
            api_type="responses" if use_responses_api else "chat/completions",
            model=model
        )
        api_messages = msg_manager(messages)
        
        # 构建请求参数
        import requests
        
        # 根据 API 类型选择端点
        if use_responses_api:
            endpoint = f"{api_base.rstrip('/')}/responses"
        else:
            endpoint = f"{api_base.rstrip('/')}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": api_messages,
            "stream": stream,
            **kwargs
        }
        
        # 添加可选参数
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        if config.max_tokens:
            payload["max_tokens"] = config.max_tokens
        
        # 发送请求
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        
        return response.json()

# 全局实例
model_manager = ModelManager()


def completion(
    model: str,
    messages: List[Any],
    tools: Optional[List[Dict]] = None,
    response_format: Optional[Dict] = None,
    stream: bool = False,
    **kwargs
) -> Dict:
    """
    Completion API 调用 (标准 chat/completions 接口)
    
    适用于大多数模型：GPT-4, Claude, Gemini, DeepSeek 等
    返回原始 OpenAI API JSON 格式响应
    自动使用 chat/completions API 端点
    
    Args:
        model: 模型名称，格式为 "provider/model" 或 "model"
               例如: "openai/gpt-4o", "gpt-4o", "anthropic/claude-3-5-sonnet-20241022"
        messages: 消息列表 (LangChain Message 对象)
        tools: 工具定义列表
        response_format: 响应格式定义
        stream: 是否流式输出
        **kwargs: 其他参数
    
    Returns:
        Dict: 原始 JSON 格式响应
        {
            "id": "chatcmpl-xxx",
            "model": "gpt-4o",
            "choices": [{"message": {"content": "...", "role": "assistant"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
    
    Example:
        >>> from langchain_core.messages import HumanMessage
        >>> response = completion(
        ...     model="openai/gpt-4o",
        ...     messages=[HumanMessage(content="Hello!")]
        ... )
        >>> print(response['choices'][0]['message']['content'])
        >>> print(response['usage'])
    """
    # 解析模型名称 (支持 provider/model 格式)
    if "/" in model:
        provider, model_name = model.split("/", 1)
        model = model_name
    
    # 自动设置使用 chat/completions API
    return model_manager.chat(
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        stream=stream,
        use_responses_api=False,  # completion() 强制使用 chat/completions
        **kwargs
    )


def response(
    model: str,
    messages: List[Any],
    tools: Optional[List[Dict]] = None,
    response_format: Optional[Dict] = None,
    stream: bool = False,
    **kwargs
) -> Dict:
    """
    Response API 调用 (新版 responses 接口，如 GPT-5)
    
    适用于使用 responses API 的模型（如 gpt-5）
    返回原始 JSON 格式响应
    自动使用 responses API 端点
    
    Args:
        model: 模型名称，格式为 "provider/model" 或 "model"
               例如: "openai/gpt-5", "gpt-5"
        messages: 消息列表 (LangChain Message 对象)
        tools: 工具定义列表
        response_format: 响应格式定义
        stream: 是否流式输出
        **kwargs: 其他参数
    
    Returns:
        Dict: 原始 JSON 格式响应
    
    Example:
        >>> from langchain_core.messages import HumanMessage
        >>> resp = response(
        ...     model="openai/gpt-5",
        ...     messages=[HumanMessage(content="Hello!")]
        ... )
        >>> print(resp['choices'][0]['message']['content'])
    """
    # 解析模型名称
    if "/" in model:
        provider, model_name = model.split("/", 1)
        model = model_name
    
    # 自动设置使用 responses API
    return model_manager.chat(
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        stream=stream,
        use_responses_api=True,  # response() 强制使用 responses API
        **kwargs
    )


if __name__ == "__main__":
    print("🚀 模型管理器测试")
    
    # 简单的测试 (如果环境中有 key)
    try:
        from langchain_core.messages import HumanMessage
        import json
        
        print("\n1️⃣ 测试 completion API...")
        messages = [HumanMessage(content="Say hello in one word")]
        resp = completion(model="openai/gpt-4o", messages=messages)
        print(f"Response type: {type(resp).__name__}")
        print(f"Response: {json.dumps(resp, indent=2, ensure_ascii=False)}")
        print(f"Content: {resp['choices'][0]['message']['content']}")
        print(f"Usage: {resp['usage']}")
        
        print("\n2️⃣ 测试 response API...")
        # 注意：需要模型支持 responses API
        # resp = response(model="openai/gpt-5", messages=messages)
        # print(f"Content: {resp['choices'][0]['message']['content']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
