"""
演示 completion() 和 response() 两种 API 调用方式
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model_manager import completion, response


def example_completion_simple():
    """示例 1: 基础 completion 调用"""
    print("=" * 60)
    print("示例 1: 基础 Completion API 调用")
    print("=" * 60)
    
    try:
        # 简单的单轮对话
        messages = [
            HumanMessage(content="用一句话介绍 Python")
        ]
        
        print("\n📤 调用模型: openai/gpt-5.1-chat")
        resp = completion(
            model="openai/gpt-5.1-chat",
            messages=messages
        )
        
        import json
        print(f"\n📥 响应类型: {type(resp).__name__}")
        print(f"📥 完整响应:")
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        
        print(f"\n💬 提取内容:")
        print(f"  内容: {resp['choices'][0]['message']['content']}")
        print(f"  Token 使用: {resp['usage']}")
        print("\n✅ 测试通过!")
        
    except Exception as e:
        print(f"❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()


def example_completion_multi_turn():
    """示例 2: 多轮对话 completion"""
    print("\n" + "=" * 60)
    print("示例 2: 多轮对话 Completion")
    print("=" * 60)
    
    try:
        messages = [
            SystemMessage(content="你是一个编程助手"),
            HumanMessage(content="什么是列表推导式？"),
            AIMessage(content="列表推导式是 Python 中创建列表的简洁语法。"),
            HumanMessage(content="给我一个例子")
        ]
        
        print("\n📤 调用模型: gpt-4o (不带 provider 前缀)")
        resp = completion(
            model="gpt-4o",  # 也可以不带 provider 前缀
            messages=messages
        )
        
        print(f"📥 响应: {resp['choices'][0]['message']['content']}\n")
        print(f"📊 Token 使用: {resp['usage']}")
        print("✅ 测试通过!")
        
    except Exception as e:
        print(f"❌ 错误: {e}\n")


def example_completion_streaming():
    """示例 3: 流式 completion"""
    print("\n" + "=" * 60)
    print("示例 3: 流式 Completion")
    print("=" * 60)
    
    try:
        messages = [
            HumanMessage(content="用三句话讲一个笑话")
        ]
        
        print("\n📤 调用模型: openai/gpt-4o (流式)")
        print("📥 流式响应:\n")
        print("-" * 60)
        
        stream = completion(
            model="openai/gpt-4o",
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            print(chunk.content, end="", flush=True)
        
        print("\n" + "-" * 60)
        print("\n✅ 测试通过!")
        
    except Exception as e:
        print(f"❌ 错误: {e}\n")


def example_completion_with_tools():
    """示例 4: 带工具调用的 completion"""
    print("\n" + "=" * 60)
    print("示例 4: Tool Call 功能测试")
    print("=" * 60)
    
    try:
        import json
        
        # 定义工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "获取指定城市的当前天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称，例如：北京、上海"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "温度单位"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "获取指定城市的当前时间",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        messages = [
            HumanMessage(content="请告诉我北京现在的天气和时间")
        ]
        
        print("\n📤 调用模型: openai/gpt-4o")
        print(f"🔧 可用工具: {len(tools)} 个")
        print("   - get_current_weather")
        print("   - get_time")
        
        resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            tools=tools
        )
        
        print(f"\n📥 完整响应:")
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        
        # 提取消息内容
        message = resp['choices'][0]['message']
        print(f"\n💬 助手消息:")
        print(f"  内容: {message.get('content', 'None')}")
        
        # 检查工具调用
        if message.get('tool_calls'):
            print(f"\n🔧 工具调用:")
            for tool_call in message['tool_calls']:
                print(f"  • ID: {tool_call['id']}")
                print(f"  • 函数: {tool_call['function']['name']}")
                print(f"  • 参数: {tool_call['function']['arguments']}")
        else:
            print("\n⚠️  没有触发工具调用")
        
        print(f"\n📊 Token 使用: {resp['usage']}")
        print("\n✅ 测试通过!")
        
    except Exception as e:
        print(f"❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()


def example_response_format():
    """示例 5: Response Format 功能测试（JSON 输出）"""
    print("\n" + "=" * 60)
    print("示例 5: Response Format - JSON 输出测试")
    print("=" * 60)
    
    try:
        import json
        
        # 测试 1: 基础 JSON 模式
        print("\n【测试 1】JSON Object 模式")
        print("-" * 60)
        
        messages = [
            SystemMessage(content="你是一个助手，请以 JSON 格式回复"),
            HumanMessage(content="给我介绍一下 Python 编程语言，包括：名称、发布年份、主要用途（列表形式）")
        ]
        
        response_format = {
            "type": "json_object"
        }
        
        print("📤 调用模型: openai/gpt-4o")
        print("📋 Response Format: json_object")
        
        resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            response_format=response_format
        )
        
        content = resp['choices'][0]['message']['content']
        print(f"\n📥 原始响应:")
        print(content)
        
        # 解析 JSON
        try:
            parsed_json = json.loads(content)
            print(f"\n✅ JSON 解析成功:")
            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
        
        print(f"\n📊 Token 使用: {resp['usage']}")
        
        # 测试 2: 结构化输出
        print("\n\n【测试 2】结构化 JSON 输出")
        print("-" * 60)
        
        messages2 = [
            SystemMessage(content="""你是一个助手，请严格按照以下 JSON 格式回复：
{
  "person": {
    "name": "姓名",
    "age": 年龄数字,
    "hobbies": ["爱好1", "爱好2"]
  },
  "summary": "一句话总结"
}"""),
            HumanMessage(content="介绍一个 30 岁的程序员，他喜欢编程和阅读")
        ]
        
        resp2 = completion(
            model="openai/gpt-4o",
            messages=messages2,
            response_format={"type": "json_object"}
        )
        
        content2 = resp2['choices'][0]['message']['content']
        print(f"📥 响应:")
        print(content2)
        
        try:
            parsed_json2 = json.loads(content2)
            print(f"\n✅ JSON 解析成功:")
            print(json.dumps(parsed_json2, indent=2, ensure_ascii=False))
            
            # 验证结构
            if 'person' in parsed_json2 and 'name' in parsed_json2['person']:
                print(f"\n✅ 结构验证通过")
                print(f"  姓名: {parsed_json2['person']['name']}")
                print(f"  年龄: {parsed_json2['person']['age']}")
                print(f"  爱好: {parsed_json2['person']['hobbies']}")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
        
        print("\n✅ 测试通过!")
        
    except Exception as e:
        print(f"❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()


def example_completion_different_providers():
    """示例 5: 不同提供商的 completion"""
    print("\n" + "=" * 60)
    print("示例 5: 不同提供商的 Completion")
    print("=" * 60)
    
    # 测试不同的提供商
    providers_models = [
        ("openai/gpt-4o", "OpenAI GPT-4o"),
        ("anthropic/claude-3-5-sonnet-20241022", "Anthropic Claude"),
        ("google/gemini-1.5-pro", "Google Gemini"),
        ("deepseek/deepseek-chat", "DeepSeek Chat"),
    ]
    
    messages = [HumanMessage(content="Say 'Hello' in one word")]
    
    for model_path, model_desc in providers_models:
        try:
            print(f"\n📤 测试模型: {model_desc}")
            print(f"   路径: {model_path}")
            
            resp = completion(
                model=model_path,
                messages=messages
            )
            
            print(f"   📥 响应: {resp.content[:50]}...")
            
        except Exception as e:
            print(f"   ❌ 错误: {str(e)[:100]}")
    
    print("\n✅ 测试完成!")


def example_response_api():
    """示例 6: Response API 调用 (GPT-5 等)"""
    print("\n" + "=" * 60)
    print("示例 6: Response API 调用")
    print("=" * 60)
    
    print("\n⚠️  注意: Response API 用于支持新版 API 的模型（如 gpt-5）")
    print("需要在 model.json 中配置 use_responses_api=true\n")
    
    try:
        messages = [
            HumanMessage(content="Hello, GPT-5!")
        ]
        
        # 注意：这个模型需要支持 responses API
        # 需要在 model.json 中配置
        print("📤 调用模型: openai/gpt-5 (需要配置)")
        
        # 取消注释以下代码进行测试（需要先配置模型）
        # resp = response(
        #     model="openai/gpt-5",
        #     messages=messages
        # )
        # print(f"📥 响应: {resp.content}")
        
        print("💡 使用方法:")
        print("   1. 在 model.json 中添加模型配置")
        print("   2. 设置 use_responses_api: true")
        print("   3. 调用 response() 函数")
        
    except Exception as e:
        print(f"❌ 错误: {e}\n")


def example_response_vs_completion():
    """示例 7: Response 和 Completion 对比"""
    print("\n" + "=" * 60)
    print("示例 7: Response API vs Completion API")
    print("=" * 60)
    
    print("\n📋 两种 API 的区别:\n")
    
    print("1️⃣ Completion API (标准 chat/completions):")
    print("   • 使用场景: 大多数模型 (GPT-4, Claude, Gemini, DeepSeek)")
    print("   • 消息格式: {\"type\": \"text\", \"text\": \"...\"}")
    print("   • 调用方式: completion(model='openai/gpt-4o', messages=...)")
    print()
    
    print("2️⃣ Response API (新版 responses):")
    print("   • 使用场景: 特定新版模型 (GPT-5 等)")
    print("   • 消息格式: {\"type\": \"input_text\", \"text\": \"...\"}")
    print("   • 调用方式: response(model='openai/gpt-5', messages=...)")
    print("   • 需要配置: use_responses_api=true")
    print()
    
    print("💡 建议:")
    print("   • 默认使用 completion() 函数")
    print("   • 只有明确需要 responses API 时才使用 response()")
    print()


def example_model_name_formats():
    """示例 8: 模型名称格式"""
    print("\n" + "=" * 60)
    print("示例 8: 模型名称格式说明")
    print("=" * 60)
    
    print("\n支持的模型名称格式:\n")
    
    print("1️⃣ 带 provider 前缀 (推荐):")
    print("   • openai/gpt-4o")
    print("   • anthropic/claude-3-5-sonnet-20241022")
    print("   • google/gemini-1.5-pro")
    print("   • deepseek/deepseek-chat")
    print()
    
    print("2️⃣ 不带 provider 前缀:")
    print("   • gpt-4o")
    print("   • claude-3-5-sonnet-20241022")
    print("   • gemini-1.5-pro")
    print()
    
    print("3️⃣ 使用别名:")
    print("   • 如果在 model.json 中配置了别名")
    print("   • 可以使用短名称，如 'gemini' -> 'gemini-pro'")
    print()


def main():
    """运行所有示例"""
    print("\n" + "🎯" * 30)
    print("Completion API 功能测试")
    print("🎯" * 30)
    
    print("\n⚠️  提示: 需要配置 API Key 才能运行实际调用")
    print("在 .env 文件中设置: API_KEY=your-key-here\n")
    
    print("📋 可用测试:")
    print("  1. 基础调用")
    print("  2. 多轮对话")
    print("  3. Response Format (JSON 输出)")
    print("  4. Tool Call (工具调用)")
    print("  5. 不同提供商")
    print("  6. API 说明（不需要 API Key）")
    print("  7. 全部测试")
    print()
    
    choice = input("请选择测试（输入数字）: ")
    
    try:
        if choice == '1':
            example_completion_simple()
        elif choice == '2':
            example_completion_multi_turn()
        elif choice == '3':
            example_response_format()
        elif choice == '4':
            example_completion_with_tools()
        elif choice == '5':
            example_completion_different_providers()
        elif choice == '6':
            example_response_vs_completion()
            example_model_name_formats()
        elif choice == '7':
            print("\n🚀 运行全部测试...\n")
            example_completion_simple()
            example_completion_multi_turn()
            example_response_format()
            example_completion_with_tools()
        else:
            print("❌ 无效选择，运行说明示例")
            example_response_vs_completion()
            example_model_name_formats()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("=" * 60)
    print("\n💡 提示:")
    print("  • Response Format: 让模型以 JSON 格式输出")
    print("  • Tool Call: 让模型调用外部工具/函数")
    print("  • 两者可以结合使用")
    print()


if __name__ == "__main__":
    main()

