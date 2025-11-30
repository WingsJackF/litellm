"""
简化版模型管理器
基于 LiteLLM 的设计思路，实现模型的注册、识别和管理功能
支持统一的 API 调用接口 (model, messages, tools, response_format)
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件（始终从当前文件所在目录加载）
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=True)  # override=True 确保 .env 优先于系统环境变量


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


@dataclass
class LLMResponse:
    """
    LLM 响应包装类
    
    根据 response_type 返回不同格式的数据：
    - "content": 只返回响应内容 (字符串)
    - "raw": 返回原始 API 响应 (完整 dict)
    
    支持两种 API 响应格式：
    - chat/completions: choices[0].message.content
    - responses API: output[0].content[0].text
    
    Example:
        >>> resp = LLMResponse(raw_response=api_result, response_type="content")
        >>> print(resp.get())  # 只返回内容字符串
        
        >>> resp = LLMResponse(raw_response=api_result, response_type="raw")
        >>> print(resp.get())  # 返回完整的原始响应
    """
    raw_response: Dict[str, Any]
    response_type: str = "content"  # "content" 或 "raw"
    
    @property
    def content(self) -> str:
        """获取响应内容（自动适配不同 API 格式）"""
        if self.raw_response is None:
            return ""
        
        try:
            # 格式1: 标准 chat/completions API
            # {"choices": [{"message": {"content": "..."}}]}
            if 'choices' in self.raw_response:
                return self.raw_response['choices'][0]['message']['content'] or ""
            
            # 格式2: OpenAI responses API (如 o1, o3, o4-mini 等)
            # {"output": [{"type": "message", "content": [{"type": "output_text", "text": "..."}]}]}
            if 'output' in self.raw_response:
                for item in self.raw_response['output']:
                    if item.get('type') == 'message':
                        for content_item in item.get('content', []):
                            if content_item.get('type') == 'output_text':
                                return content_item.get('text', '')
            
            return ""
        except (KeyError, IndexError, TypeError):
            return ""
    
    @property
    def raw(self) -> Dict[str, Any]:
        """获取原始响应"""
        return self.raw_response
    
    @property
    def usage(self) -> Optional[Dict[str, int]]:
        """获取 token 使用情况"""
        if self.raw_response is None:
            return None
        return self.raw_response.get('usage')
    
    @property
    def model(self) -> Optional[str]:
        """获取实际使用的模型名称"""
        if self.raw_response is None:
            return None
        return self.raw_response.get('model')
    
    def get(self) -> Union[str, Dict[str, Any]]:
        """根据 response_type 返回对应数据"""
        if self.response_type == "raw":
            return self.raw_response
        return self.content
    
    def __str__(self) -> str:
        """字符串表示，返回内容"""
        return self.content
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] if self.content else ""
        return f"LLMResponse(response_type='{self.response_type}', content='{content_preview}...')"


class ModelManager:
    """
    模型管理器 - 管理和识别不同的 LLM 模型
    
    功能：
    1. 注册和管理模型配置
    2. 统一的 API 调用接口 (chat)
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
        
        # 提供商列表
        self.providers: List[str] = [
            "openai", "anthropic", "google", "deepseek"
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
            "gpt-5": "openai",
            "o3": "openai",
            "o3-deep-research": "openai",  # Deep Research 模型
            "computer-use-preview": "openai",  # Computer Use 模型
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
        return config

    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        if model in self.model_aliases:
            model = self.model_aliases[model]
        return self.models.get(model)

    def _get_api_key(self, provider: str, config_key: Optional[str] = None, use_provider_specific: bool = False) -> Optional[str]:
        """
        获取 API Key
        
        Args:
            provider: 提供商名称
            config_key: 配置中的 key
            use_provider_specific: 是否优先使用提供商特定的环境变量（用于 responses API）
        """
        # 提供商特定的环境变量映射
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        
        if use_provider_specific:
            # responses API: 优先使用提供商特定的环境变量
            # 1. 提供商特定的环境变量
            provider_key = os.getenv(env_keys.get(provider, ""))
            if provider_key:
                return provider_key
            # 2. 配置中的 key
            if config_key:
                return config_key
            # 3. 统一的 API_KEY 作为后备
            return os.getenv("API_KEY")
        else:
            # completion API: 优先使用统一的 API_KEY（代理）
            # 1. 统一的 API_KEY
            unified_key = os.getenv("API_KEY")
            if unified_key:
                return unified_key
            # 2. 配置中的 key
            if config_key:
                return config_key
            # 3. 提供商特定的环境变量
            return os.getenv(env_keys.get(provider, ""))

    def _get_api_base(self, provider: str, config_base: Optional[str] = None, use_provider_specific: bool = False) -> Optional[str]:
        """
        获取 API Base
        
        Args:
            provider: 提供商名称
            config_base: 配置中的 base
            use_provider_specific: 是否优先使用提供商特定的环境变量（用于 responses API）
        """
        # 提供商特定的环境变量
        env_var = f"{provider.upper()}_API_BASE"
        
        if use_provider_specific:
            # responses API: 优先使用提供商特定的环境变量
            # 1. 提供商特定的环境变量
            provider_base = os.getenv(env_var)
            if provider_base:
                return provider_base
            # 2. 配置中的 base
            if config_base:
                return config_base
            # 3. 默认的提供商 API base
            return self.provider_api_bases.get(provider)
        else:
            # completion API: 优先使用统一的 BASE_URL（代理）
            # 1. 统一的 BASE_URL
            unified_base = os.getenv("BASE_URL")
            if unified_base:
                return unified_base
            # 2. 提供商特定的环境变量
            env_base = os.getenv(env_var)
            if env_base:
                return env_base
            # 3. 配置中的 base
            return config_base

    def chat(
        self, 
        model: str, 
        messages: List[Any], 
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
        use_responses_api: Optional[bool] = None,
        use_provider_api: bool = False,
        provider: Optional[str] = None,
        **kwargs
    ) -> Union[Dict, Any]:
        """
        统一 API 调用接口 - 使用 OpenAI SDK
        
        Args:
            model: 模型名称
            messages: 消息列表 (HumanMessage, AIMessage, SystemMessage 对象)
            tools: 工具定义列表
            response_format: 响应格式定义
            stream: 是否流式输出
            use_responses_api: 是否使用 responses API（None 时使用配置文件设置）
            use_provider_api: 是否使用厂商原始 API（True 时优先使用 PROVIDER_API_KEY/BASE）
            provider: 提供商名称（如 openai, anthropic, google 等）
            **kwargs: 其他参数
        
        Returns:
            Dict: 原始 OpenAI API 格式的 JSON 响应（非流式）
            Stream: 流式响应对象（流式）
            格式: {
                "id": "chatcmpl-xxx",
                "model": "gpt-4o",
                "choices": [{"message": {"content": "...", "role": "assistant"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
        """
        # 获取模型配置，如果没有找到则使用传入的 provider 或默认 openai
        config = self.get_model_config(model)
        if config is None:
            default_provider = provider or "openai"
            config = ModelConfig(model, default_provider)
        
        # 确定使用哪种 API：优先使用参数，其次使用配置
        if use_responses_api is None:
            use_responses_api = config.use_responses_api
        
        # 根据 API 类型决定获取 key 和 base 的优先级
        # use_provider_api=True: 强制使用厂商原始 API (PROVIDER_API_KEY, PROVIDER_API_BASE)
        # responses API: 优先使用提供商特定的环境变量 (OPENAI_API_KEY, OPENAI_API_BASE)
        # completion API: 优先使用统一的代理 (API_KEY, BASE_URL)
        use_provider_specific = use_provider_api or use_responses_api
        api_key = self._get_api_key(config.provider, config.api_key, use_provider_specific=use_provider_specific)
        api_base = self._get_api_base(config.provider, config.api_base, use_provider_specific=use_provider_specific)
        
        # 转换 Messages 为 API 格式
        from message_manager import MessageManager
        msg_manager = MessageManager(
            api_type="responses" if use_responses_api else "chat/completions",
            model=model
        )
        api_messages = msg_manager(messages)
        
        # 创建 OpenAI 客户端（添加超时设置）
        timeout = kwargs.pop('timeout', 120)  # 默认 120 秒超时
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )
        
        # 根据 API 类型选择不同的调用方式
        if use_responses_api:
            # 使用 responses API（如 GPT-5, o1, o3 等）
            # 端点: /responses
            params = {
                "model": model,
                "input": api_messages,  # responses API 使用 input 而不是 messages
                "stream": stream,
                **kwargs
            }
            
            # 添加可选参数
            if tools:
                params["tools"] = tools
            if config.max_tokens:
                params["max_tokens"] = config.max_tokens
            
            # 调试输出
            print(f"🔄 Calling API: {api_base}/responses")
            print(f"   Model: {model}, Timeout: {timeout}s")
            
            # 调用 responses API
            response = client.responses.create(**params)
        else:
            # 使用标准 chat/completions API
            params = {
                "model": model,
                "messages": api_messages,
                "stream": stream,
                **kwargs
            }
            
            # 添加可选参数
            if tools:
                params["tools"] = tools
            if response_format:
                params["response_format"] = response_format
            if config.max_tokens:
                params["max_tokens"] = config.max_tokens
            
            # 调试输出
            print(f"🔄 Calling API: {api_base}/chat/completions")
            print(f"   Model: {model}, Timeout: {timeout}s")
            
            # 调用 chat/completions API
            response = client.chat.completions.create(**params)
        
        # 流式响应直接返回
        if stream:
            return response
        
        # 非流式响应转换为 dict
        return response.model_dump()

# 全局实例
model_manager = ModelManager()


def completion(
    model: str,
    messages: List[Any],
    tools: Optional[List[Dict]] = None,
    response_format: Optional[Dict] = None,
    stream: bool = False,
    response_type: str = "raw",
    use_provider_api: bool = False,
    **kwargs
) -> Union[str, Dict, LLMResponse]:
    """
    Completion API 调用 (标准 chat/completions 接口)
    
    适用于大多数模型：GPT-4, Claude, Gemini, DeepSeek 等
    自动使用 chat/completions API 端点
    
    Args:
        model: 模型名称，格式为 "provider/model" 或 "model"
               例如: "openai/gpt-4o", "gpt-4o", "anthropic/claude-3-5-sonnet-20241022"
        messages: 消息列表 (HumanMessage, AIMessage, SystemMessage 对象)
        tools: 工具定义列表
        response_format: 响应格式定义
        stream: 是否流式输出
        response_type: 响应类型
            - "content": 只返回内容字符串
            - "raw": 返回原始 API 响应 dict (默认)
        use_provider_api: 是否使用厂商原始 API（True 时使用 PROVIDER_API_KEY/BASE）
        **kwargs: 其他参数
    
    Returns:
        根据 response_type 返回:
        - "content": str (响应内容)
        - "raw": Dict (原始 JSON 格式响应)
    
    Example:
        >>> from message_manager import HumanMessage
        >>> # 获取原始响应
        >>> resp = completion(model="gpt-4o", messages=[HumanMessage(content="Hello!")])
        >>> print(resp['choices'][0]['message']['content'])
        
        >>> # 只获取内容
        >>> content = completion(model="gpt-4o", messages=[HumanMessage(content="Hello!")], response_type="content")
        >>> print(content)  # 直接输出字符串
        
        >>> # 使用厂商原始 API（Computer Use 等场景）
        >>> resp = completion(model="anthropic/claude", messages=msgs, use_provider_api=True)
    """
    # 解析模型名称 (支持 provider/model 格式)
    provider = None
    if "/" in model:
        provider, model_name = model.split("/", 1)
        model = model_name
    
    # 自动设置使用 chat/completions API
    raw_response = model_manager.chat(
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        stream=stream,
        use_responses_api=False,  # completion() 强制使用 chat/completions
        use_provider_api=use_provider_api,  # 是否使用厂商原始 API
        provider=provider,  # 传递解析出的 provider
        **kwargs
    )
    
    # 流式响应直接返回
    if stream:
        return raw_response
    
    # 根据 response_type 返回对应格式
    llm_response = LLMResponse(raw_response=raw_response, response_type=response_type)
    return llm_response.get()


def response(
    model: str,
    messages: List[Any],
    tools: Optional[List[Dict]] = None,
    response_format: Optional[Dict] = None,
    stream: bool = False,
    response_type: str = "raw",
    use_provider_api: bool = False,
    **kwargs
) -> Union[str, Dict, LLMResponse]:
    """
    Response API 调用 (新版 responses 接口，如 GPT-5)
    
    适用于使用 responses API 的模型（如 gpt-5）
    自动使用 responses API 端点
    
    Args:
        model: 模型名称，格式为 "provider/model" 或 "model"
               例如: "openai/gpt-5", "gpt-5"
        messages: 消息列表 (HumanMessage, AIMessage, SystemMessage 对象)
        tools: 工具定义列表
        response_format: 响应格式定义
        stream: 是否流式输出
        response_type: 响应类型
            - "content": 只返回内容字符串
            - "raw": 返回原始 API 响应 dict (默认)
        use_provider_api: 是否使用厂商原始 API（True 时使用 PROVIDER_API_KEY/BASE）
        **kwargs: 其他参数
    
    Returns:
        根据 response_type 返回:
        - "content": str (响应内容)
        - "raw": Dict (原始 JSON 格式响应)
    
    Example:
        >>> from message_manager import HumanMessage
        >>> # 获取原始响应
        >>> resp = response(model="gpt-5", messages=[HumanMessage(content="Hello!")])
        >>> print(resp['choices'][0]['message']['content'])
        
        >>> # 只获取内容
        >>> content = response(model="gpt-5", messages=[HumanMessage(content="Hello!")], response_type="content")
        >>> print(content)  # 直接输出字符串
    """
    # 解析模型名称
    provider = None
    if "/" in model:
        provider, model_name = model.split("/", 1)
        model = model_name
    
    # 自动设置使用 responses API
    raw_response = model_manager.chat(
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        stream=stream,
        use_responses_api=True,  # response() 强制使用 responses API
        use_provider_api=use_provider_api,  # 是否使用厂商原始 API
        provider=provider,  # 传递解析出的 provider
        **kwargs
    )
    
    # 流式响应直接返回
    if stream:
        return raw_response
    
    # 根据 response_type 返回对应格式
    llm_response = LLMResponse(raw_response=raw_response, response_type=response_type)
    return llm_response.get()


if __name__ == "__main__":
    from message_manager import HumanMessage
    import json as json_module
    from datetime import datetime
    
    # ============================================
    # 测试输出日志类（同时输出到控制台和文件）
    # ============================================
    class TestLogger:
        def __init__(self, output_file: str = "test_results.md"):
            self.output_file = Path(__file__).parent / output_file
            self.lines = []
            
        def log(self, message: str = ""):
            """输出到控制台并记录"""
            print(message)
            self.lines.append(message)
        
        def save(self):
            """保存到 md 文件（覆盖模式）"""
            with open(self.output_file, 'w', encoding='utf-8') as f:
                # 添加标题和时间戳
                f.write("# 模型管理器测试结果\n\n")
                f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                
                # 写入所有日志
                for line in self.lines:
                    # 转换格式为 markdown
                    if line.startswith("="*50):
                        f.write("\n---\n\n")
                    elif "测试" in line and not line.startswith(" "):
                        f.write(f"## {line}\n\n")
                    elif line.startswith("📝") or line.startswith("📋") or line.startswith("🖼️") or line.startswith("🚀"):
                        f.write(f"### {line}\n\n")
                    elif line.startswith("   "):
                        # 结果行
                        f.write(f"```\n{line.strip()}\n```\n\n")
                    elif line.startswith("⚠️") or line.startswith("❌"):
                        f.write(f"> {line}\n\n")
                    elif line.startswith("📁"):
                        f.write(f"**{line}**\n\n")
                    else:
                        f.write(f"{line}\n\n")
            
            print(f"\n📄 测试结果已保存到: {self.output_file}")
    
    # 初始化日志
    logger = TestLogger()
    log = logger.log
    
    log("🚀 模型管理器测试")
    
    # 通用测试消息
    simple_messages = [HumanMessage(content="Say hello in one word")]
    format_messages = [HumanMessage(content="生成一个虚构人物的信息，包含姓名、年龄和爱好。")]
    
    # 通用图片消息（网络 URL）
    image_messages = [
        HumanMessage(content=[
            {"type": "text", "text": "这张图片里有什么？请用中文简短描述。"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://q7.itc.cn/q_70/images03/20250219/6c6b4e75e7e6412999a728d67ba7a8d2.jpeg"
                }
            }
        ])
    ]
    
    # 通用结构化输出格式
    structured_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "人物姓名"},
                    "age": {"type": "integer", "description": "年龄"},
                    "hobbies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "爱好列表"
                    }
                },
                "required": ["name", "age", "hobbies"],
                "additionalProperties": False
            }
        }
    }
    
    # JSON 格式（兼容不支持 json_schema 的模型）
    json_format = {"type": "json_object"}
    
    def parse_person(content: str):
        """解析人物信息 JSON"""
        try:
            person = json_module.loads(content)
            return f"姓名={person.get('name')}, 年龄={person.get('age')}, 爱好={person.get('hobbies')}"
        except:
            return content
    
    # ============================================
    # 1️⃣ OpenAI 测试
    # ============================================
    log("\n" + "="*50)
    log("1️⃣ OpenAI 模型测试")
    log("="*50)
    
    try:
        log("\n📝 基本问答测试completions (gpt-4o)...")
        resp = completion(model="openai/gpt-4o", messages=simple_messages, response_type="content")
        log(f"   Response: {resp}")

        log("\n📝 基本问答测试response (gpt-4o)...")
        resp = response(model="openai/gpt-4o", messages=simple_messages, response_type="content")
        log(f"   Response: {resp}")
        
        log("\n📋 结构化输出测试 (gpt-4o)...")
        resp = completion(
            model="openai/gpt-4o", 
            messages=format_messages, 
            response_format=structured_format,
            response_type="content"
        )
        log(f"   Structured: {parse_person(resp)}")
        
        log("\n🖼️ 图片理解测试-网络URL (gpt-5)...")
        resp = response(model="openai/gpt-5", messages=image_messages, response_type="raw")
        log(f"   图片描述: {resp}")
        
        log("\n🚀 Response API 测试 (gpt-5)...")
        resp = response(model="openai/gpt-5", messages=simple_messages, response_type="content")
        log(f"   Response: {resp}")
        
        log("\n🔬 Deep Research 流式测试 (o3-deep-research)...")
        # Deep Research 需要配置 web_search_preview 工具
        research_messages = [
            HumanMessage(content="请深入研究并总结：量子计算的基本原理是什么？它与经典计算有什么根本区别？")
        ]
        # Deep Research 必须配置工具：web_search_preview, mcp, 或 file_search
        deep_research_tools = [
            {"type": "web_search_preview"}  # 启用网页搜索能力
        ]
        
        # 流式输出
        print("   [流式输出开始]")
        stream_resp = response(
            model="openai/o3-deep-research",
            messages=research_messages,
            tools=deep_research_tools,
            stream=True,  # 启用流式输出
            timeout=600
        )
        
        # 处理流式响应
        full_content = ""
        for event in stream_resp:
            # 根据事件类型处理
            if hasattr(event, 'type'):
                if event.type == 'response.output_text.delta':
                    # 文本增量
                    delta = event.delta if hasattr(event, 'delta') else ""
                    print(delta, end="", flush=True)
                    full_content += delta
                elif event.type == 'response.completed':
                    # 响应完成
                    print("\n   [流式输出结束]")
            elif hasattr(event, 'choices'):
                # chat/completions 流式格式
                for choice in event.choices:
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        delta = choice.delta.content or ""
                        print(delta, end="", flush=True)
                        full_content += delta
        
        log(f"   Research Result (总长度: {len(full_content)} 字符)")
        
        log("\n🖥️ Computer Use 测试 (computer-use-preview)...")
        log("   💡 提示: 使用 computer_use.py 模块进行真实自动化")
        log("   运行: python computer_use.py")
        
        # 简单测试：只发送一次请求看响应
        computer_tool = {
            "type": "computer_use_preview",
            "display_width": 1920,
            "display_height": 1080,
            "environment": "browser",
        }
        
        resp = response(
            model="openai/computer-use-preview",
            messages=[HumanMessage(content="截取当前屏幕")],
            tools=[computer_tool],
            response_type="raw",
            timeout=60,
            truncation="auto"
        )
        
        if resp and 'output' in resp:
            for item in resp['output']:
                if item.get('type') == 'computer_call':
                    action = item.get('action', {})
                    log(f"   ✅ Computer Action: {action.get('type', 'unknown')}")
        else:
            log(f"   ⚠️ 响应: {str(resp)[:200]}...")
        
    except Exception as e:
        log(f"   ❌ OpenAI 测试失败: {e}")
    
    # ============================================
    # 🖼️ 本地图片测试（独立测试块）
    # ============================================
    log("\n" + "="*50)
    log("🖼️ 本地图片上传测试")
    log("="*50)
    
    try:
        # 测试本地图片路径（请替换为实际存在的图片路径）
        local_image_path = "./test_image/img1.webp"
        
        # 检查文件是否存在
        import os
        if os.path.exists(local_image_path):
            log(f"\n📁 找到本地图片: {local_image_path}")
            
            # 直接使用 HumanMessage 构造本地图片消息
            local_image_msgs = [
                HumanMessage(content=[
                    {"type": "text", "text": "这张图片里有什么？请用中文简短描述。"},
                    {"type": "image_url", "image_url": {"url": local_image_path}}
                ])
            ]
            
            log("\n🖼️ 本地图片测试 (gpt-4o - completion)...")
            resp = completion(model="openai/gpt-4o", messages=local_image_msgs, response_type="content")
            log(f"   图片描述: {resp}")
            
            log("\n🖼️ 本地图片测试 (gemini-2.5-pro - completion)...")
            resp = completion(model="gemini-2.5-pro", messages=local_image_msgs, response_type="content")
            log(f"   图片描述: {resp}")
        else:
            log(f"\n⚠️ 本地图片不存在: {local_image_path}")
            log("   请创建测试图片或修改 local_image_path 变量")
            
    except Exception as e:
        log(f"   ❌ 本地图片测试失败: {e}")
    
    # ============================================
    # 2️⃣ Qwen (通义千问) 测试
    # ============================================
    log("\n" + "="*50)
    log("2️⃣ Qwen (通义千问) 模型测试")
    log("="*50)
    
    try:
        log("\n📝 基本问答测试 (qwen-plus)...")
        resp = completion(model="qwen-plus", messages=simple_messages, response_type="content")
        log(f"   Response: {resp}")
        
        log("\n📋 结构化输出测试 (qwen-plus)...")
        # Qwen 使用 json_object 格式
        qwen_format_messages = [HumanMessage(content="生成一个虚构人物的JSON信息，包含name(姓名)、age(年龄)和hobbies(爱好数组)字段。只输出JSON。")]
        resp = completion(
            model="qwen-plus", 
            messages=qwen_format_messages, 
            response_format=structured_format,
            response_type="content"
        )
        log(f"   Structured: {parse_person(resp)}")
        
        log("\n🖼️ 图片理解测试 (qwen3-vl-plus)...")
        resp = completion(model="qwen3-vl-plus", messages=image_messages, response_type="content")
        log(f"   图片描述: {resp}")
        
    except Exception as e:
        log(f"   ❌ Qwen 测试失败: {e}")
    
    # ============================================
    # 3️⃣ DeepSeek 测试
    # ============================================
    log("\n" + "="*50)
    log("3️⃣ DeepSeek 模型测试")
    log("="*50)
    
    try:
        log("\n📝 基本问答测试 (deepseek-v3.2-exp)...")
        resp = completion(model="deepseek-v3.2-exp", messages=simple_messages, response_type="content")
        log(f"   Response: {resp}")
        
        log("\n📋 结构化输出测试 (deepseek-v3.2-exp)...")
        deepseek_format_messages = [HumanMessage(content="生成一个虚构人物的JSON信息，包含name(姓名)、age(年龄)和hobbies(爱好数组)字段。只输出JSON，不要其他内容。")]
        resp = completion(
            model="deepseek-v3.2-exp", 
            messages=deepseek_format_messages, 
            response_format=structured_format,
            response_type="content"
        )
        log(f"   Structured: {parse_person(resp)}")
        
    except Exception as e:
        log(f"   ❌ DeepSeek 测试失败: {e}")
    
    # ============================================
    # 4️⃣ Claude (Anthropic) 测试
    # ============================================
    log("\n" + "="*50)
    log("4️⃣ Claude (Anthropic) 模型测试")
    log("="*50)
    
    try:
        log("\n📝 基本问答测试 (claude-sonnet-4-5-20250929)...")
        resp = completion(model="claude-sonnet-4-5-20250929", messages=simple_messages, response_type="content")
        log(f"   Response: {resp}")
        
        log("\n📋 结构化输出测试 (claude-sonnet-4-5-20250929)...")
        claude_format_messages = [HumanMessage(content="生成一个虚构人物的JSON信息，包含name(姓名)、age(年龄)和hobbies(爱好数组)字段。只输出纯JSON，不要markdown代码块。")]
        resp = completion(
            model="claude-sonnet-4-5-20250929", 
            messages=claude_format_messages, 
            response_type="content",
            response_format=structured_format
        )
        log(f"   Structured: {parse_person(resp)}")
        
    except Exception as e:
        log(f"   ❌ Claude 测试失败: {e}")
    
    # ============================================
    # 5️⃣ Gemini (Google) 测试
    # ============================================
    log("\n" + "="*50)
    log("5️⃣ Gemini (Google) 模型测试")
    log("="*50)
    
    try:
        log("\n📝 基本问答测试 (gemini-2.5-pro)...")
        resp = completion(model="gemini-2.5-pro", messages=simple_messages, response_type="content")
        log(f"   Response: {resp}")
        
        log("\n📋 结构化输出测试 (gemini-2.5-pro)...")
        gemini_format_messages = [HumanMessage(content="生成一个虚构人物的JSON信息，包含name(姓名)、age(年龄)和hobbies(爱好数组)字段。只输出纯JSON。")]
        resp = completion(
            model="gemini-2.5-pro", 
            messages=gemini_format_messages, 
            response_format=structured_format,
            response_type="content"
        )
        log(f"   Structured: {parse_person(resp)}")
        
        log("\n🖼️ 图片理解测试 (gemini-2.5-pro)...")
        resp = completion(model="gemini-2.5-pro", messages=image_messages, response_type="content")
        log(f"   图片描述: {resp}")
        
    except Exception as e:
        log(f"   ❌ Gemini 测试失败: {e}")
    
    # ============================================
    # 6. Computer Use 测试（简化版：只测试点击位置）
    # ============================================
    log("\n" + "="*50)
    log("6. Computer Use 测试 (点击位置测试)")
    log("="*50)
    
    try:
        from PIL import Image
        
        # 测试图片路径（1024x768 的截图）
        test_screenshot = "./test_image/screenshot.png"
        
        if os.path.exists(test_screenshot):
            log(f"\n找到测试截图: {test_screenshot}")
            
            # 验证图片尺寸
            with Image.open(test_screenshot) as img:
                width, height = img.size
                log(f"   图片尺寸: {width}x{height}")
            
            # 使用 HumanMessage 构造消息（本地图片会自动转换为 base64）
            computer_messages = [
                HumanMessage(content=[
                    {"type": "text", "text": "请在屏幕上找到并点击 model.json 文件"},
                    {"type": "image_url", "image_url": {"url": test_screenshot}}
                ])
            ]
            
            # Computer Use 工具配置
            computer_tool = {
                "type": "computer_use_preview",
                "display_width": 1024,
                "display_height": 768,
                "environment": "mac"
            }
            
            log("\nComputer Use 点击测试 (computer-use-preview)...")
            log("   任务: 点击 model.json 文件")
            
            # 使用 response() 函数调用 API（使用厂商原始 API）
            resp = response(
                model="openai/computer-use-preview",
                messages=computer_messages,
                tools=[computer_tool],
                response_type="raw",
                use_provider_api=True,  # 使用 OpenAI 原始 API
                truncation="auto",
                timeout=60
            )
            
            # 解析响应，获取点击位置
            if resp and 'output' in resp:
                for item in resp['output']:
                    if item.get('type') == 'computer_call':
                        action = item.get('action', {})
                        action_type = action.get('type', 'unknown')
                        
                        if action_type == 'click':
                            x = action.get('x', 0)
                            y = action.get('y', 0)
                            button = action.get('button', 'left')
                            log(f"   模型返回点击动作:")
                            log(f"      位置: ({x}, {y})")
                            log(f"      按钮: {button}")
                        elif action_type == 'screenshot':
                            log(f"   模型请求截图")
                        else:
                            log(f"   动作类型: {action_type}, 详情: {action}")
                    elif item.get('type') == 'message':
                        for content in item.get('content', []):
                            if content.get('type') == 'output_text':
                                log(f"   模型消息: {content.get('text', '')[:100]}")
            else:
                log(f"   响应格式异常: {str(resp)[:200]}...")
                
        else:
            log(f"\n测试截图不存在: {test_screenshot}")
            log("   请先截取一张 1024x768 的屏幕截图并保存到 test_image/screenshot.png")
            
    except Exception as e:
        log(f"   Computer Use 测试失败: {e}")
    
    # ============================================
    # 7. Anthropic Computer Use 测试
    # ============================================
    # log("\n" + "="*50)
    # log("7. Anthropic Computer Use 测试 (点击位置测试)")
    # log("="*50)
    
    # try:
    #     from PIL import Image
        
    #     # 测试图片路径（1024x768 的截图）
    #     test_screenshot = "./test_image/screenshot.png"
        
    #     if os.path.exists(test_screenshot):
    #         log(f"\n找到测试截图: {test_screenshot}")
            
    #         # 验证图片尺寸
    #         with Image.open(test_screenshot) as img:
    #             width, height = img.size
    #             log(f"   图片尺寸: {width}x{height}")
            
    #         # 使用 HumanMessage 构造消息（本地图片自动转换为 base64）
    #         computer_messages = [
    #             HumanMessage(content=[
    #                 {"type": "text", "text": "请在屏幕上找到并点击 model.json 文件"},
    #                 {"type": "image_url", "image_url": {"url": test_screenshot}}
    #             ])
    #         ]
            
    #         # Anthropic Computer Use 工具配置
    #         computer_tool = {
    #             "type": "computer_20250124",
    #             "name": "computer",
    #             "display_width_px": 1024,
    #             "display_height_px": 768,
    #             "display_number": 1,
    #         }
            
    #         log("\nAnthropic Computer Use 点击测试...")
    #         log("   任务: 点击 model.json 文件")
            
    #         # 使用 completion() 函数调用 API（使用厂商原始 API）
    #         resp = completion(
    #             model="anthropic/computer-use-2025-11-24",
    #             messages=computer_messages,
    #             tools=[computer_tool],
    #             response_type="raw",
    #             use_provider_api=True,  # 使用 Anthropic 原始 API
    #             timeout=60
    #         )
            
    #         # 调试：打印原始响应
    #         log(f"   原始响应: {json_module.dumps(resp, ensure_ascii=False, indent=2)[:500] if resp else 'None'}...")
            
    #         # 解析响应，获取点击位置
    #         if resp and 'choices' in resp:
    #             message = resp['choices'][0].get('message', {})
    #             tool_calls = message.get('tool_calls', [])
                
    #             if tool_calls:
    #                 for tool_call in tool_calls:
    #                     func_name = tool_call.get('function', {}).get('name', '')
    #                     if func_name == 'computer':
    #                         args = json_module.loads(tool_call['function'].get('arguments', '{}'))
    #                         action = args.get('action', '')
                            
    #                         if action == 'left_click':
    #                             coords = args.get('coordinate', [0, 0])
    #                             log(f"   模型返回点击动作:")
    #                             log(f"      位置: ({coords[0]}, {coords[1]})")
    #                             log(f"      动作: left_click")
    #                         elif action == 'screenshot':
    #                             log(f"   模型请求截图")
    #                         else:
    #                             log(f"   动作类型: {action}, 详情: {args}")
    #             else:
    #                 # 没有 tool_calls，查看文本内容
    #                 content = message.get('content', '')
    #                 if content:
    #                     log(f"   模型消息: {content[:200]}")
    #         elif resp:
    #             log(f"   响应格式: {list(resp.keys()) if isinstance(resp, dict) else type(resp)}")
    #         else:
    #             log(f"   响应为空")
                
    #     else:
    #         log(f"\n测试截图不存在: {test_screenshot}")
    #         log("   请先截取一张 1024x768 的屏幕截图并保存到 test_image/screenshot.png")
            
    # except Exception as e:
    #     log(f"   Anthropic Computer Use 测试失败: {e}")
    
    log("\n" + "="*50)
    log("测试完成")
    log("="*50)
    
    # 保存测试结果到 md 文件
    logger.save()
