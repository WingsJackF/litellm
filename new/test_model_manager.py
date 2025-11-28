"""
模型管理器测试文件
演示如何使用 ModelManager 进行模型调用
"""

from model_manager import model_manager
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def list_available_models():
    """列出所有可用的模型"""
    print("=" * 60)
    print("📋 可用模型列表")
    print("=" * 60)
    
    # 1. 显示已知模型
    print("\n🔹 已知模型 (Known Models):")
    for model_name, provider in model_manager.known_models.items():
        config = model_manager.get_model_config(model_name)
        supports = []
        if config:
            if config.supports_vision:
                supports.append("视觉")
            if config.supports_functions:
                supports.append("函数")
            if config.supports_streaming:
                supports.append("流式")
        support_str = f" [{', '.join(supports)}]" if supports else ""
        print(f"  • {model_name:30s} (提供商: {provider}){support_str}")
    
    # 2. 显示自定义模型
    custom_models = [name for name in model_manager.models.keys() 
                     if name not in model_manager.known_models]
    if custom_models:
        print("\n🔹 自定义模型 (Custom Models):")
        for model_name in custom_models:
            config = model_manager.get_model_config(model_name)
            print(f"  • {model_name:30s} (提供商: {config.provider})")
    
    # 3. 显示模型别名
    if model_manager.model_aliases:
        print("\n🔹 模型别名 (Model Aliases):")
        for alias, real_name in model_manager.model_aliases.items():
            print(f"  • {alias} → {real_name}")
    
    # 4. 显示支持的提供商
    print("\n🔹 支持的提供商 (Providers):")
    for provider in model_manager.providers:
        api_base = model_manager.provider_api_bases.get(provider, "N/A")
        print(f"  • {provider:15s} - {api_base}")
    
    print("\n" + "=" * 60)
    print(f"✅ 总计: {len(model_manager.models)} 个模型\n")


def test_simple_chat():
    """测试用例 1: 简单对话"""
    print("=" * 60)
    print("🧪 测试用例 1: 简单对话")
    print("=" * 60)
    
    try:
        # 创建消息 (使用 LangChain Message 对象)
        messages = [
            HumanMessage(content="你好，请用一句话介绍你自己。")
        ]
        
        # 调用模型
        model_name = "gpt-4o"  # 可以修改为其他模型
        print(f"\n📤 正在调用模型: {model_name}")
        print(f"💬 消息: {messages[0].content}")
        
        response = model_manager.chat(
            model=model_name,
            messages=messages
        )
        print(response)
        print(f"\n📥 响应:")
        print(f"  类型: {type(response).__name__}")
        print(f"  内容: {response.content}")
        
        # 获取 token 使用情况
        client = model_manager.get_model(model_name)
        if hasattr(client, 'callbacks') and client.callbacks:
            for callback in client.callbacks:
                if hasattr(callback, 'input_tokens'):
                    print(f"\n📊 Token 使用统计:")
                    print(f"  输入 tokens: {callback.input_tokens}")
                    print(f"  输出 tokens: {callback.output_tokens}")
                    print(f"  总计 tokens: {callback.total_tokens}")
                    print(f"  耗时: {callback.total_duration:.2f}秒")
        
        print("\n✅ 测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_multi_turn_chat():
    """测试用例 2: 多轮对话"""
    print("\n" + "=" * 60)
    print("🧪 测试用例 2: 多轮对话")
    print("=" * 60)
    
    try:
        # 创建多轮对话消息
        messages = [
            SystemMessage(content="你是一个有用的助手。"),
            HumanMessage(content="Python 中如何定义一个函数？"),
            AIMessage(content="在 Python 中，使用 `def` 关键字定义函数。"),
            HumanMessage(content="能给一个例子吗？")
        ]
        
        model_name = "gpt-4o"
        print(f"\n📤 正在调用模型: {model_name}")
        print(f"💬 对话轮数: {len(messages)} 条消息")
        
        response = model_manager.chat(
            model=model_name,
            messages=messages
        )
        print(response)
        print(f"\n📥 响应:")
        print(f"  {response}")
        
        print("\n✅ 测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_streaming_chat():
    """测试用例 3: 流式输出"""
    print("\n" + "=" * 60)
    print("🧪 测试用例 3: 流式输出")
    print("=" * 60)
    
    try:
        messages = [
            HumanMessage(content="用三句话讲一个小故事。")
        ]
        
        model_name = "gpt-4o"
        print(f"\n📤 正在调用模型: {model_name} (流式)")
        print(f"💬 消息: {messages[0].content}")
        print(f"\n📥 流式响应:")
        print("-" * 60)
        
        stream = model_manager.chat(
            model=model_name,
            messages=messages,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            content = chunk.content
            print(content, end="", flush=True)
            full_response += content
        
        print("\n" + "-" * 60)
        print(f"✅ 接收完成，总长度: {len(full_response)} 字符")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_with_tools():
    """测试用例 4: 使用工具 (Tool Calling)"""
    print("\n" + "=" * 60)
    print("🧪 测试用例 4: 工具调用")
    print("=" * 60)
    
    try:
        # 定义工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称，例如：北京、上海"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]
        
        messages = [
            HumanMessage(content="北京今天天气怎么样？")
        ]
        
        model_name = "gpt-4o"
        print(f"\n📤 正在调用模型: {model_name}")
        print(f"💬 消息: {messages[0].content}")
        print(f"🔧 工具数量: {len(tools)}")
        
        response = model_manager.chat(
            model=model_name,
            messages=messages,
            tools=tools
        )
        print(response)
        print(f"\n📥 响应:")
        print(f"  内容: {response}")
        
        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"  🔧 工具调用:")
            for tool_call in response.tool_calls:
                print(f"    • 函数: {tool_call.get('name', 'N/A')}")
                print(f"    • 参数: {tool_call.get('args', {})}")
        
        print("\n✅ 测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_register_custom_model():
    """测试用例 5: 注册自定义模型"""
    print("\n" + "=" * 60)
    print("🧪 测试用例 5: 注册自定义模型")
    print("=" * 60)
    
    try:
        # 注册一个自定义模型
        custom_model_name = "my-custom-model"
        config = model_manager.register_model(
            model_name=custom_model_name,
            provider="openai",
            api_base="https://api.custom.com/v1",
            api_key="sk-custom-key-xxx",
            supports_streaming=True,
            supports_functions=True,
            max_tokens=4096
        )
        
        print(f"\n✅ 成功注册自定义模型:")
        print(f"  模型名称: {config.model_name}")
        print(f"  提供商: {config.provider}")
        print(f"  API Base: {config.api_base}")
        print(f"  最大 tokens: {config.max_tokens}")
        
        # 验证是否可以获取配置
        retrieved_config = model_manager.get_model_config(custom_model_name)
        if retrieved_config:
            print(f"\n✅ 模型配置已持久化到 model.json")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主测试函数"""
    print("\n" + "🚀" * 30)
    print("ModelManager 完整测试套件")
    print("🚀" * 30 + "\n")
    
    # 列出所有可用模型
    list_available_models()
    
    # 运行测试用例
    # 注意: 需要配置相应的 API Key 才能运行
    print("\n⚠️  提示: 以下测试需要配置 API Key (在 .env 文件中设置 API_KEY 或 OPENAI_API_KEY)")
    print("如果没有配置，测试将失败。\n")
    
    user_input = input("是否运行测试用例？(y/n): ")
    if user_input.lower() == 'y':
        test_simple_chat()
        test_multi_turn_chat()
        test_streaming_chat()
        test_with_tools()
        test_register_custom_model()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

