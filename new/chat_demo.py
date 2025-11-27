#!/usr/bin/env python
"""
实际聊天演示 - 真实调用 API
展示如何使用 ModelManager 和 MessageManager 构建真实的聊天应用
"""

import os
from model_manager import model_manager
from message_manager import MessageManager
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def _generate_model_descriptions():
    """
    从 ModelManager 动态生成模型列表（包括所有已注册的模型）
    """
    available_models = {}
    idx = 1
    
    # 收集所有模型和它们的提供商
    all_models = {}
    
    # 1. 从 known_models 获取预定义模型
    for model_name, provider in model_manager.known_models.items():
        all_models[model_name] = provider
    
    # 2. 从 models 获取通过 register_model 注册的模型（包括自定义模型）
    for model_name, config in model_manager.models.items():
        if model_name not in all_models:  # 避免重复
            all_models[model_name] = config.provider
    
    # 3. 按提供商分组
    models_by_provider = {}
    for model_name, provider in all_models.items():
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(model_name)
    
    # 4. 按提供商排序（优先显示常用提供商，其他按字母顺序）
    provider_priority = ["openai", "anthropic", "deepseek", "google", "groq", "mistral"]
    sorted_providers = []
    
    # 先添加优先提供商（如果存在）
    for provider in provider_priority:
        if provider in models_by_provider:
            sorted_providers.append(provider)
    
    # 再添加其他提供商（按字母顺序）
    other_providers = sorted([p for p in models_by_provider.keys() if p not in provider_priority])
    sorted_providers.extend(other_providers)
    
    # 5. 生成模型列表
    for provider in sorted_providers:
        # 对每个提供商的模型按字母排序
        for model_name in sorted(models_by_provider[provider]):
            description = f"{provider.upper()} - {model_name}"
            available_models[str(idx)] = (model_name, description)
            idx += 1
    
    return available_models


# 动态生成可用模型配置（会在每次调用时刷新）
def get_available_models():
    """获取当前所有可用模型（包括新注册的）"""
    return _generate_model_descriptions()

# 初始化时生成一次（向后兼容）
AVAILABLE_MODELS = get_available_models()


def show_models_list(show_provider: bool = True):
    """
    显示所有可用模型列表
    
    Args:
        show_provider: 是否显示提供商信息
    """
    # 获取最新的模型列表（包括新注册的）
    models = get_available_models()
    
    print("\n📋 可用模型列表:")
    print("-" * 70)
    
    if show_provider:
        # 按提供商分组显示
        current_provider = None
        for key, (model, desc) in models.items():
            provider = model_manager.known_models.get(model, "unknown")
            
            if provider != current_provider:
                if current_provider is not None:
                    print()
                print(f"  【{provider.upper()}】")
                current_provider = provider
            
            print(f"    {key:2}. {desc}")
    else:
        # 简单列表
        for key, (model, desc) in models.items():
            print(f"  {key:2}. {desc}")
    
    print("-" * 70)


def select_model(prompt: str = "请选择模型", default: str = "1", show_provider: bool = False) -> tuple:
    """
    让用户选择模型
    
    Args:
        prompt: 提示信息
        default: 默认选项
        show_provider: 是否显示提供商分组
        
    Returns:
        tuple: (model_name, model_description)
    """
    # 获取最新的模型列表
    models = get_available_models()
    
    print(f"\n{prompt}:")
    
    if show_provider:
        show_models_list(show_provider=True)
    else:
        print("-" * 70)
        for key, (model, desc) in models.items():
            print(f"  {key:2}. {desc}")
        print("-" * 70)
    
    choice = input(f"\n请输入选项 (1-{len(models)}，默认 {default}): ").strip() or default
    
    if choice not in models:
        print(f"❌ 无效选择，使用默认选项 {default}")
        choice = default
    
    model_name, model_desc = models[choice]
    
    # 获取模型的提供商信息
    provider = model_manager.known_models.get(model_name, "unknown")
    print(f"✅ 已选择: {model_desc} (提供商: {provider})\n")
    
    return model_name, model_desc


# 方式 1: 使用 requests 直接调用（最基础）
def chat_with_requests(model: str, messages: list):
    """使用 requests 库直接调用 API"""
    import requests
    
    # 获取模型配置
    model_name, provider, api_key, api_base = model_manager.get_llm_provider(model)
    
    print(f"\n📡 调用模型: {model_name}")
    print(f"   提供商: {provider}")
    print(f"   API Base: {api_base}")
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "stream": False
    }
    
    # 发送请求
    response = requests.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API 调用失败: {response.status_code} - {response.text}")


# 方式 2: 使用 OpenAI SDK（推荐）
def chat_with_openai_sdk(model: str, messages: list):
    """使用 OpenAI SDK 调用 API（兼容所有 OpenAI 格式的 API）"""
    from openai import OpenAI
    
    # 获取模型配置
    model_name, provider, api_key, api_base = model_manager.get_llm_provider(model)
    
    # 创建客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    print(f"\n📡 调用模型: {model_name}")
    print(f"   提供商: {provider}")
    print(f"   API Base: {api_base}")
    
    # 发送请求
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content


# 方式 3: 使用 LiteLLM（最灵活）
def chat_with_litellm(model: str, messages: list):
    """使用 LiteLLM 调用 API（支持所有主流模型提供商）"""
    import litellm
    
    # 获取模型配置
    model_name, provider, api_key, api_base = model_manager.get_llm_provider(model)
    
    print(f"\n📡 调用模型: {model_name}")
    print(f"   提供商: {provider}")
    print(f"   API Base: {api_base}")
    
    # 发送请求
    response = litellm.completion(
        model=f"{provider}/{model_name}",
        messages=messages,
        api_key=api_key,
        api_base=api_base,
        temperature=0.7
    )
    
    return response.choices[0].message.content


# 完整的聊天机器人类
class ChatBot:
    """完整的聊天机器人实现"""
    
    def __init__(self, model: str, system_prompt: str = None, method: str = "openai"):
        """
        初始化聊天机器人
        
        Args:
            model: 模型名称（如 "gpt-4", "groq/llama-3.1-8b" 等）
            system_prompt: 系统提示词
            method: 调用方式 ("requests", "openai", "litellm")
        """
        self.model = model
        self.method = method
        self.message_manager = MessageManager(
            system_prompt=system_prompt or "你是一个有帮助的 AI 助手。"
        )
        
        # 验证模型配置
        try:
            model_name, provider, api_key, api_base = model_manager.get_llm_provider(model)
            print(f"✅ 聊天机器人已初始化")
            print(f"   模型: {model_name}")
            print(f"   提供商: {provider}")
            print(f"   API Base: {api_base}")
            print(f"   调用方式: {method}")
        except Exception as e:
            print(f"❌ 模型配置失败: {e}")
            raise
    
    def chat(self, user_input: str, images: list = None) -> str:
        """
        与机器人对话
        
        Args:
            user_input: 用户输入
            images: 可选的图像 URL 列表（用于多模态对话）
            
        Returns:
            助手的回复
        """
        # 添加用户消息
        if images:
            self.message_manager.add_multimodal_message(
                role="user",
                text=user_input,
                images=images
            )
        else:
            self.message_manager.add_user_message(user_input)
        
        # 获取消息历史
        messages = self.message_manager.get_messages(format="dict")
        
        # 调用 API
        try:
            if self.method == "requests":
                response = chat_with_requests(self.model, messages)
            elif self.method == "openai":
                response = chat_with_openai_sdk(self.model, messages)
            elif self.method == "litellm":
                response = chat_with_litellm(self.model, messages)
            else:
                raise ValueError(f"未知的调用方式: {self.method}")
            
            # 检查响应是否为空
            if not response or not response.strip():
                error_msg = "❌ API 返回空响应"
                print(error_msg)
                # 移除刚才添加的用户消息
                self.message_manager.pop_last_message()
                return error_msg
            
            # 添加助手回复
            self.message_manager.add_assistant_message(response)
            
            return response
            
        except Exception as e:
            error_msg = f"❌ API 调用失败: {str(e)}"
            print(error_msg)
            # 移除刚才添加的用户消息
            self.message_manager.pop_last_message()
            return error_msg
    
    def chat_stream(self, user_input: str):
        """流式对话（实时返回）"""
        from openai import OpenAI
        
        # 添加用户消息
        self.message_manager.add_user_message(user_input)
        messages = self.message_manager.get_messages(format="dict")
        
        # 获取模型配置
        model_name, provider, api_key, api_base = model_manager.get_llm_provider(self.model)
        
        # 创建客户端
        client = OpenAI(api_key=api_key, base_url=api_base)
        
        print(f"\n🤖 助手: ", end="", flush=True)
        
        try:
            # 流式请求
            full_response = ""
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                temperature=0.7
            )
            
            for chunk in stream:
                # 更健壮的错误处理 - 检查 chunk 结构
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        print(content, end="", flush=True)
                        full_response += content
            
            print()  # 换行
            
            # 检查响应是否为空
            if not full_response.strip():
                error_msg = "（API 返回空响应）"
                print(error_msg)
                # 移除刚才添加的用户消息，因为对话失败了
                self.message_manager.pop_last_message()
                return error_msg
            
            # 添加助手回复到历史
            self.message_manager.add_assistant_message(full_response)
            
            return full_response
            
        except Exception as e:
            error_msg = f"\n❌ API 调用失败: {str(e)}"
            print(error_msg)
            # 移除刚才添加的用户消息，因为对话失败了
            self.message_manager.pop_last_message()
            return error_msg
    
    def print_history(self):
        """打印对话历史"""
        self.message_manager.print_history()
    
    def clear_history(self):
        """清空对话历史"""
        self.message_manager.clear_history()
    
    def export_chat(self, filepath: str):
        """导出对话历史"""
        self.message_manager.export_history(filepath)


# 示例 1: 基础对话
def demo_basic_chat():
    print("=" * 70)
    print("📱 示例 1: 基础对话")
    print("=" * 70)
    
    # 让用户选择模型
    selected_model, _ = select_model("选择要使用的模型", default="1")
    
    # 创建聊天机器人（使用你的代理 API）
    bot = ChatBot(
        model=selected_model,
        system_prompt="你是一个友好的 AI 助手，请用简洁的语言回答。",
        method="openai"  # 使用 OpenAI SDK
    )
    
    # 进行对话
    print("\n👤 用户: 你好！")
    response = bot.chat("你好！")
    print(f"🤖 助手: {response}")
    
    print("\n👤 用户: 用 Python 写一个 Hello World")
    response = bot.chat("用 Python 写一个 Hello World")
    print(f"🤖 助手: {response}")
    
    # 查看历史
    print("\n" + "=" * 70)
    bot.print_history()


# 示例 2: 流式对话
def demo_stream_chat():
    print("\n" + "=" * 70)
    print("⚡ 示例 2: 流式对话（实时返回）")
    print("=" * 70)
    
    # 让用户选择模型
    selected_model, _ = select_model("选择要使用的模型", default="1")
    
    bot = ChatBot(
        model=selected_model,
        system_prompt="你是一个有帮助的编程助手。",
        method="openai"
    )
    
    print("\n👤 用户: 解释一下什么是装饰器")
    bot.chat_stream("解释一下什么是装饰器")


# 示例 3: 多模态对话
def demo_multimodal_chat():
    print("\n" + "=" * 70)
    print("🎨 示例 3: 多模态对话（文本 + 图像）")
    print("=" * 70)
    
    bot = ChatBot(
        model="gpt-4o",  # 需要支持视觉的模型
        system_prompt="你是一个可以理解图像的 AI 助手。",
        method="openai"
    )
    
    print("\n👤 用户: [发送图片] 这张图片里有什么？")
    response = bot.chat(
        "这张图片里有什么？",
        images=["https://example.com/image.jpg"]
    )
    print(f"🤖 助手: {response}")


# 示例 4: 模型对比
def demo_model_comparison():
    """同时向多个模型提问，对比回答"""
    print("\n" + "=" * 70)
    print("⚖️  示例 4: 模型对比")
    print("=" * 70)
    
    # 获取最新模型列表
    models = get_available_models()
    
    print("\n📝 选择要对比的模型（至少选择 2 个，最多 4 个）")
    print("   输入模型编号，用空格或逗号分隔，例如: 1 5 6")
    print()
    
    # 显示模型列表
    for key, (model, desc) in models.items():
        print(f"  {key:2}. {desc}")
    
    # 获取用户选择
    choices = input("\n请输入模型编号: ").strip()
    choices = choices.replace(",", " ").split()
    
    if len(choices) < 2:
        print("❌ 至少需要选择 2 个模型")
        return
    
    if len(choices) > 4:
        print("⚠️  最多支持 4 个模型，只使用前 4 个")
        choices = choices[:4]
    
    # 创建多个聊天机器人
    bots = []
    for choice in choices:
        if choice in models:
            model_name, model_desc = models[choice]
            try:
                bot = ChatBot(
                    model=model_name,
                    system_prompt="你是一个有帮助的 AI 助手，请用简洁的语言回答。",
                    method="openai"
                )
                bots.append((model_name, model_desc, bot))
                print(f"✅ 已加载: {model_desc}")
            except Exception as e:
                print(f"❌ 加载失败 {model_desc}: {e}")
    
    if len(bots) < 2:
        print("❌ 可用模型不足 2 个")
        return
    
    print(f"\n✅ 已准备 {len(bots)} 个模型进行对比\n")
    
    # 获取用户问题
    question = input("请输入你的问题: ").strip()
    
    if not question:
        print("❌ 问题不能为空")
        return
    
    print("\n" + "=" * 70)
    print(f"📢 问题: {question}")
    print("=" * 70)
    
    # 依次调用每个模型
    for i, (model_name, model_desc, bot) in enumerate(bots, 1):
        print(f"\n【模型 {i}: {model_desc}】")
        print("-" * 70)
        
        try:
            response = bot.chat(question)
            print(response)
        except Exception as e:
            print(f"❌ 调用失败: {e}")
        
        print("-" * 70)
    
    print("\n✅ 对比完成！")


# 示例 5: 交互式聊天
def interactive_chat():
    print("\n" + "=" * 70)
    print("💬 交互式聊天 (输入 'quit' 退出)")
    print("=" * 70)
    
    # 让用户选择模型
    selected_model, model_desc = select_model("可用模型列表", default="1")
    
    # 创建聊天机器人
    bot = ChatBot(
        model=selected_model,
        system_prompt="你是一个友好、有帮助的 AI 助手。",
        method="openai"
    )
    
    print("\n✅ 聊天机器人已准备就绪！开始对话吧...")
    print("💡 提示: 输入 /help 查看可用命令\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 再见！")
                break
            
            # 特殊命令
            if user_input == "/help":
                print("\n📚 可用命令:")
                print("-" * 70)
                print("  /help          - 显示此帮助信息")
                print("  /models        - 显示所有可用模型（按提供商分组）")
                print("  /switch        - 切换模型")
                print("  /current       - 显示当前模型信息")
                print("  /stats         - 显示模型统计信息")
                print("  /history       - 显示对话历史")
                print("  /clear         - 清空对话历史")
                print("  /export <file> - 导出对话历史到文件")
                print("  quit/exit/q    - 退出程序")
                print("-" * 70)
                
                # 获取最新模型数量
                current_models = get_available_models()
                print(f"\n💡 当前共支持 {len(current_models)} 个模型")
                continue
            
            if user_input == "/models":
                # 获取最新模型列表
                current_models = get_available_models()
                
                print("\n📋 可用模型列表:")
                print("-" * 70)
                
                # 按提供商分组显示
                current_provider = None
                for key, (model, desc) in current_models.items():
                    provider = model_manager.known_models.get(model, "unknown")
                    
                    if provider != current_provider:
                        if current_provider is not None:
                            print()
                        print(f"  【{provider.upper()}】")
                        current_provider = provider
                    
                    current = " ← 当前使用" if model == selected_model else ""
                    print(f"    {key:2}. {desc}{current}")
                
                print("-" * 70)
                print(f"\n💡 提示: 输入 /switch 可以切换模型")
                continue
            
            if user_input == "/switch":
                # 获取最新模型列表
                current_models = get_available_models()
                
                print("\n🔄 切换模型:")
                print("-" * 70)
                
                # 按提供商分组显示
                current_provider = None
                for key, (model, desc) in current_models.items():
                    provider = model_manager.known_models.get(model, "unknown")
                    
                    if provider != current_provider:
                        if current_provider is not None:
                            print()
                        print(f"  【{provider.upper()}】")
                        current_provider = provider
                    
                    current = " ← 当前使用" if model == selected_model else ""
                    print(f"    {key:2}. {desc}{current}")
                
                print("-" * 70)
                
                new_choice = input(f"\n请选择新模型 (1-{len(current_models)}): ").strip()
                
                if new_choice in current_models:
                    selected_model, model_desc = current_models[new_choice]
                    provider = model_manager.known_models.get(selected_model, "unknown")
                    print(f"\n🔄 正在切换到: {model_desc} (提供商: {provider})")
                    
                    # 创建新的聊天机器人（保留历史）
                    old_messages = bot.message_manager.messages
                    bot = ChatBot(
                        model=selected_model,
                        system_prompt="你是一个友好、有帮助的 AI 助手。",
                        method="openai"
                    )
                    bot.message_manager.messages = old_messages
                    print("✅ 模型切换成功！\n")
                else:
                    print("❌ 无效选择\n")
                continue
            
            if user_input == "/stats":
                # 获取最新模型列表
                current_models = get_available_models()
                
                print("\n📊 模型统计信息:")
                print("-" * 70)
                
                # 统计各提供商的模型数量
                provider_counts = {}
                for model in model_manager.known_models.values():
                    provider_counts[model] = provider_counts.get(model, 0) + 1
                
                print(f"  总模型数: {len(current_models)}")
                print(f"  支持的提供商数: {len(provider_counts)}")
                print()
                print("  各提供商模型数:")
                for provider, count in sorted(provider_counts.items()):
                    print(f"    • {provider.title()}: {count} 个")
                
                print("-" * 70 + "\n")
                continue
            
            if user_input == "/current":
                try:
                    model_name, provider, api_key, api_base = model_manager.get_llm_provider(selected_model)
                    print(f"\n📊 当前模型信息:")
                    print("-" * 70)
                    print(f"  模型名称: {model_name}")
                    print(f"  提供商: {provider}")
                    print(f"  API Base: {api_base}")
                    print(f"  API Key: {'已设置 ✅' if api_key else '未设置 ❌'}")
                    print()
                    print(f"  对话统计:")
                    print(f"    • 消息总数: {len(bot.message_manager.messages)}")
                    print(f"    • Token 估算: ~{bot.message_manager.count_tokens_estimate()}")
                    msg_stats = bot.message_manager.count_messages()
                    print(f"    • 用户消息: {msg_stats.get('user', 0)}")
                    print(f"    • 助手消息: {msg_stats.get('assistant', 0)}")
                    print("-" * 70 + "\n")
                except Exception as e:
                    print(f"❌ 获取模型信息失败: {e}\n")
                continue
            
            if user_input == "/history":
                bot.print_history()
                continue
            
            if user_input == "/clear":
                bot.clear_history()
                print("✅ 历史已清空\n")
                continue
            
            if user_input.startswith("/export "):
                filepath = user_input.split(" ", 1)[1]
                bot.export_chat(filepath)
                continue
            
            # 正常对话
            # 使用流式输出
            bot.chat_stream(user_input)
            
            print()  # 空行
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def main():
    """主函数"""
    print("\n" + "🎯 " * 35)
    print("   实际聊天演示 - 使用真实 API")
    print("🎯 " * 35 + "\n")
    
    # 检查环境变量
    if not os.getenv("API_KEY"):
        print("⚠️  警告: 未检测到 API_KEY 环境变量")
        print("   请在 .env 文件中设置:")
        print("   BASE_URL=https://your-proxy-api.com/v1")
        print("   API_KEY=your-api-key-here")
        print()
        return
    
    print("✅ 环境配置已加载")
    print(f"   BASE_URL: {os.getenv('BASE_URL', '未设置')}")
    print(f"   API_KEY: {'已设置' if os.getenv('API_KEY') else '未设置'}")
    print()
    
    # 选择运行模式
    print("请选择运行模式:")
    print("  1. 基础对话演示")
    print("  2. 流式对话演示")
    print("  3. 多模态对话演示（需要支持视觉的模型）")
    print("  4. 模型对比（同时向多个模型提问）")
    print("  5. 交互式聊天（推荐）⭐")
    print("  6. 运行所有演示")
    
    try:
        choice = input("\n请输入选项 (1-6，默认 5): ").strip() or "5"
        
        if choice == "1":
            demo_basic_chat()
        elif choice == "2":
            demo_stream_chat()
        elif choice == "3":
            demo_multimodal_chat()
        elif choice == "4":
            demo_model_comparison()
        elif choice == "5":
            interactive_chat()
        elif choice == "6":
            demo_basic_chat()
            demo_stream_chat()
            demo_model_comparison()
            # demo_multimodal_chat()  # 需要支持视觉的模型
        else:
            print("❌ 无效选项")
            
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")


if __name__ == "__main__":
    main()

