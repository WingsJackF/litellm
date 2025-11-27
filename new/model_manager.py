"""
简化版模型管理器
基于 LiteLLM 的设计思路，实现模型的注册、识别和管理功能
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

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


class ModelManager:
    """
    模型管理器 - 管理和识别不同的 LLM 模型
    
    功能：
    1. 注册和管理模型配置
    2. 根据模型名称自动识别提供商
    3. 获取模型的 API 配置信息
    4. 支持模型别名映射
    """
    
    def __init__(self, model_file: str = "model.json"):
        """
        初始化模型管理器
        
        Args:
            model_file: 模型配置文件路径，默认为 model.json
        """
        # 模型配置文件路径
        self.model_file = Path(__file__).parent / model_file
        
        # 已注册的模型配置
        self.models: Dict[str, ModelConfig] = {}
        
        # 提供商列表
        self.providers: List[str] = [
            "openai", "anthropic", "azure", "cohere", "replicate",
            "huggingface", "groq", "mistral", "deepseek", "together_ai",
            "perplexity", "anyscale", "bedrock", "vertex_ai", "google", "ollama"
        ]
        
        # 提供商默认 API Base URL 映射
        self.provider_api_bases: Dict[str, str] = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
            "google": "https://generativelanguage.googleapis.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "mistral": "https://api.mistral.ai/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "together_ai": "https://api.together.xyz/v1",
            "perplexity": "https://api.perplexity.ai",
            "anyscale": "https://api.endpoints.anyscale.com/v1",
            "ollama": "http://localhost:11434/v1",
        }
        
        # 已知模型到提供商的映射
        self.known_models: Dict[str, str] = {
            # OpenAI 模型
            "gpt-4": "openai",
            "gpt-4-turbo": "openai",
            "gpt-4o": "openai",
            "gpt-3.5-turbo": "openai",
            "gpt-4o-mini": "openai",
            
            # Anthropic 模型
            "claude-opus-4-5-20251101": "anthropic",
           
            
            
            # DeepSeek 模型
            "deepseek-chat": "deepseek"
        }
        
        # 模型别名映射
        self.model_aliases: Dict[str, str] = {}
        
        # 初始化默认模型配置
        self._initialize_default_models()
        
        # 从 JSON 文件加载自定义模型
        self._load_from_json()
    
    def _initialize_default_models(self):
        """初始化默认支持的模型配置"""
        for model_name, provider in self.known_models.items():
            config = ModelConfig(
                model_name=model_name,
                provider=provider,
                api_base=self.provider_api_bases.get(provider),
                supports_vision="gpt-4" in model_name or "claude-3" in model_name
            )
            self.models[model_name] = config
    
    def _load_from_json(self):
        """从 JSON 文件加载自定义模型配置"""
        try:
            if self.model_file.exists():
                with open(self.model_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 加载自定义模型
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
                            max_tokens=model_data.get('max_tokens')
                        )
                        self.models[model_data['model_name']] = config
                
                # 加载别名
                if 'aliases' in data:
                    self.model_aliases = data['aliases']
                
                print(f"✅ 已从 {self.model_file.name} 加载 {len(data.get('custom_models', []))} 个自定义模型")
        except json.JSONDecodeError:
            print(f"⚠️  {self.model_file.name} 格式错误，跳过加载")
        except Exception as e:
            print(f"⚠️  加载模型配置失败: {e}")
    
    def _save_to_json(self):
        """保存自定义模型配置到 JSON 文件"""
        try:
            # 只保存自定义模型（不在 known_models 中的）
            custom_models = []
            for model_name, config in self.models.items():
                if model_name not in self.known_models:
                    model_data = {
                        'model_name': config.model_name,
                        'provider': config.provider,
                        'api_base': config.api_base,
                        'api_key': config.api_key,
                        'default_params': config.default_params,
                        'supports_streaming': config.supports_streaming,
                        'supports_functions': config.supports_functions,
                        'supports_vision': config.supports_vision,
                        'max_tokens': config.max_tokens
                    }
                    custom_models.append(model_data)
            
            data = {
                'custom_models': custom_models,
                'aliases': self.model_aliases
            }
            
            with open(self.model_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # print(f"✅ 已保存 {len(custom_models)} 个自定义模型到 {self.model_file.name}")
        except Exception as e:
            print(f"❌ 保存模型配置失败: {e}")
    
    def register_model(
        self,
        model_name: str,
        provider: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ModelConfig:
        """
        注册新模型（会自动保存到 JSON 文件）
        
        Args:
            model_name: 模型名称
            provider: 提供商名称
            api_base: API 基础 URL
            api_key: API 密钥
            **kwargs: 其他配置参数
            
        Returns:
            ModelConfig: 模型配置对象
        """
        config = ModelConfig(
            model_name=model_name,
            provider=provider,
            api_base=api_base or self.provider_api_bases.get(provider),
            api_key=api_key,
            **kwargs
        )
        self.models[model_name] = config
        
        # 保存到 JSON 文件
        self._save_to_json()
        
        print(f"✅ 已注册模型: {model_name} (提供商: {provider})")
        return config
    
    def add_model_alias(self, alias: str, actual_model: str):
        """
        添加模型别名（会自动保存到 JSON 文件）
        
        Args:
            alias: 别名
            actual_model: 实际模型名称
        """
        self.model_aliases[alias] = actual_model
        
        # 保存到 JSON 文件
        self._save_to_json()
        
        print(f"✅ 已添加别名: {alias} -> {actual_model}")
    
    def get_llm_provider(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        根据模型名称识别提供商和配置（模仿 LiteLLM 的 get_llm_provider）
        
        Args:
            model: 模型名称
            api_base: 可选的 API base URL
            api_key: 可选的 API key
            
        Returns:
            Tuple[model_name, provider, api_key, api_base]
        """
        # 检查是否是别名
        if model in self.model_aliases:
            model = self.model_aliases[model]
        
        # 1. 检查模型名称中是否包含提供商前缀 (如 "groq/llama-3.1-8b")
        if "/" in model:
            parts = model.split("/", 1)
            if parts[0] in self.providers:
                provider = parts[0]
                model_name = parts[1]
                
                # 获取该提供商的默认配置
                final_api_base = api_base or self._get_api_base_from_env(provider) or self.provider_api_bases.get(provider)
                final_api_key = api_key or self._get_api_key_from_env(provider)
                
                return model_name, provider, final_api_key, final_api_base
        
        # 2. 检查是否是已注册的模型
        if model in self.models:
            config = self.models[model]
            final_api_base = api_base or self._get_api_base_from_env(config.provider) or config.api_base
            final_api_key = api_key or config.api_key or self._get_api_key_from_env(config.provider)
            
            return config.model_name, config.provider, final_api_key, final_api_base
        
        # 3. 检查是否在已知模型列表中
        if model in self.known_models:
            provider = self.known_models[model]
            final_api_base = api_base or self._get_api_base_from_env(provider) or self.provider_api_bases.get(provider)
            final_api_key = api_key or self._get_api_key_from_env(provider)
            
            return model, provider, final_api_key, final_api_base
        
        # 4. 如果提供了 api_base，尝试从中识别提供商
        if api_base:
            for provider, base_url in self.provider_api_bases.items():
                if base_url in api_base:
                    final_api_key = api_key or self._get_api_key_from_env(provider)
                    return model, provider, final_api_key, api_base
        
        # 5. 无法识别，抛出错误
        raise ValueError(
            f"❌ 无法识别模型: {model}\n"
            f"支持的格式:\n"
            f"  1. 使用提供商前缀: 'groq/llama-3.1-8b'\n"
            f"  2. 已知模型名称: {list(self.known_models.keys())[:5]}...\n"
            f"  3. 注册新模型: manager.register_model(...)"
        )
    
    def _get_api_key_from_env(self, provider: str) -> Optional[str]:
        """从环境变量获取 API 密钥（优先使用统一的 API_KEY）"""
        # 优先使用统一的 API_KEY
        unified_key = os.environ.get("API_KEY")
        if unified_key:
            return unified_key
        
        # 如果没有统一的 API_KEY，则使用特定提供商的 key
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "together_ai": "TOGETHER_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "anyscale": "ANYSCALE_API_KEY",
        }
        
        env_var = env_keys.get(provider)
        if env_var:
            return os.environ.get(env_var)
        return None
    
    def _get_api_base_from_env(self, provider: str) -> Optional[str]:
        """从环境变量获取 API Base URL（优先使用统一的 BASE_URL）"""
        # 优先使用统一的 BASE_URL
        unified_base = os.environ.get("BASE_URL")
        if unified_base:
            return unified_base
        
        # 如果没有统一的 BASE_URL，则使用特定提供商的 base URL
        env_var = f"{provider.upper()}_API_BASE"
        return os.environ.get(env_var)
    
    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        if model in self.model_aliases:
            model = self.model_aliases[model]
        return self.models.get(model)
    
    def list_models(self) -> List[str]:
        """列出所有已注册的模型"""
        return list(self.models.keys())
    
    def list_providers(self) -> List[str]:
        """列出所有支持的提供商"""
        return self.providers
    
    def remove_model(self, model: str):
        """移除已注册的模型（会自动保存到 JSON 文件）"""
        if model in self.models:
            # 不能删除预定义模型
            if model in self.known_models:
                print(f"⚠️  不能删除预定义模型: {model}")
                return
            
            del self.models[model]
            
            # 保存到 JSON 文件
            self._save_to_json()
            
            print(f"✅ 已移除模型: {model}")
        else:
            print(f"⚠️  模型不存在: {model}")
    
    def update_model(self, model: str, **kwargs):
        """更新模型配置（会自动保存到 JSON 文件）"""
        if model not in self.models:
            raise ValueError(f"模型 {model} 不存在")
        
        config = self.models[model]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 保存到 JSON 文件
        self._save_to_json()
        
        print(f"✅ 已更新模型配置: {model}")
        return config
    
    def get_model_info(self, model: str) -> Dict:
        """获取模型详细信息"""
        try:
            model_name, provider, api_key, api_base = self.get_llm_provider(model)
            
            config = self.get_model_config(model)
            
            info = {
                "model_name": model_name,
                "provider": provider,
                "api_base": api_base,
                "has_api_key": api_key is not None,
            }
            
            if config:
                info.update({
                    "supports_streaming": config.supports_streaming,
                    "supports_functions": config.supports_functions,
                    "supports_vision": config.supports_vision,
                    "max_tokens": config.max_tokens,
                })
            
            return info
        except Exception as e:
            return {"error": str(e)}
    
    def print_model_info(self, model: str):
        """打印模型信息"""
        info = self.get_model_info(model)
        print(f"\n📊 模型信息: {model}")
        print("=" * 50)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("=" * 50)


# 全局模型管理器实例
model_manager = ModelManager()


if __name__ == "__main__":
    # 示例用法
    print("🚀 模型管理器示例\n")
    
    # 1. 使用已知模型
    print("1️⃣ 识别已知模型:")
    model, provider, key, base = model_manager.get_llm_provider("gpt-4")
    print(f"   模型: {model}, 提供商: {provider}, API Base: {base}\n")
    
    # 2. 使用提供商前缀
    print("2️⃣ 使用提供商前缀:")
    model, provider, key, base = model_manager.get_llm_provider("groq/llama-3.1-8b")
    print(f"   模型: {model}, 提供商: {provider}, API Base: {base}\n")
    
    # 3. 注册自定义模型
    print("3️⃣ 注册自定义模型:")
    model_manager.register_model(
        model_name="my-custom-model",
        provider="openai",
        api_base="https://my-custom-endpoint.com/v1",
        supports_vision=True
    )
    
    # 4. 添加别名
    print("\n4️⃣ 添加模型别名:")
    model_manager.add_model_alias("gpt4", "gpt-4")
    model, provider, key, base = model_manager.get_llm_provider("gpt4")
    print(f"   别名 'gpt4' -> 实际模型: {model}\n")
    
    # 5. 查看模型详细信息
    print("5️⃣ 查看模型详细信息:")
    model_manager.print_model_info("gpt-4")
    
    # 6. 列出所有模型
    print(f"\n6️⃣ 已注册模型数量: {len(model_manager.list_models())}")
    print(f"   支持的提供商: {', '.join(model_manager.list_providers()[:5])}...")

