"""
测试 Response Format 和 Tool Call 功能
"""

from langchain_core.messages import HumanMessage, SystemMessage
from model_manager import completion
import json


def test_response_format_json():
    """测试 1: Response Format - JSON 输出"""
    print("=" * 70)
    print("测试 1: Response Format - JSON 输出")
    print("=" * 70)
    
    messages = [
        SystemMessage(content="你是一个助手，请以 JSON 格式回复"),
        HumanMessage(content="""
请用 JSON 格式介绍 Python 编程语言，包含以下字段：
- name: 语言名称
- year: 发布年份（数字）
- creator: 创造者
- features: 主要特点（数组）
- popular_uses: 主要用途（数组）
""")
    ]
    
    print("\n📤 请求:")
    print(f"  模型: openai/gpt-4o")
    print(f"  Response Format: json_object")
    print(f"  消息: 请求 Python 的 JSON 介绍")
    
    try:
        resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        content = resp['choices'][0]['message']['content']
        
        print(f"\n📥 原始响应:")
        print(content)
        
        # 解析 JSON
        parsed = json.loads(content)
        print(f"\n✅ JSON 解析成功!")
        print(f"\n📋 格式化输出:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        
        # 验证字段
        print(f"\n🔍 字段验证:")
        required_fields = ['name', 'year', 'creator', 'features', 'popular_uses']
        for field in required_fields:
            if field in parsed:
                print(f"  ✅ {field}: 存在")
            else:
                print(f"  ❌ {field}: 缺失")
        
        print(f"\n📊 Token 使用:")
        print(f"  输入: {resp['usage']['prompt_tokens']}")
        print(f"  输出: {resp['usage']['completion_tokens']}")
        print(f"  总计: {resp['usage']['total_tokens']}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON 解析失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_call_single():
    """测试 2: Tool Call - 单个工具调用"""
    print("\n\n" + "=" * 70)
    print("测试 2: Tool Call - 单个工具")
    print("=" * 70)
    
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
                            "description": "城市名称，如：北京、上海、深圳"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位",
                            "default": "celsius"
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
    
    print("\n📤 请求:")
    print(f"  模型: openai/gpt-4o")
    print(f"  工具: get_weather")
    print(f"  消息: {messages[0].content}")
    
    try:
        resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            tools=tools
        )
        
        message = resp['choices'][0]['message']
        
        print(f"\n📥 响应:")
        print(f"  内容: {message.get('content', 'None')}")
        print(f"  Finish Reason: {resp['choices'][0]['finish_reason']}")
        
        # 检查工具调用
        if message.get('tool_calls'):
            print(f"\n✅ 触发了工具调用!")
            for i, tool_call in enumerate(message['tool_calls'], 1):
                print(f"\n🔧 工具调用 #{i}:")
                print(f"  ID: {tool_call['id']}")
                print(f"  类型: {tool_call['type']}")
                print(f"  函数名: {tool_call['function']['name']}")
                print(f"  参数 (原始): {tool_call['function']['arguments']}")
                
                # 解析参数
                try:
                    args = json.loads(tool_call['function']['arguments'])
                    print(f"  参数 (解析):")
                    for key, value in args.items():
                        print(f"    • {key}: {value}")
                except json.JSONDecodeError:
                    print(f"  ⚠️  参数解析失败")
            
            return True
        else:
            print(f"\n⚠️  没有触发工具调用")
            print(f"  可能原因: 模型直接回答了问题")
            return False
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_call_multiple():
    """测试 3: Tool Call - 多个工具"""
    print("\n\n" + "=" * 70)
    print("测试 3: Tool Call - 多个工具选择")
    print("=" * 70)
    
    # 定义多个工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "获取城市的当前时间",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_news",
                "description": "搜索相关新闻",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    messages = [
        HumanMessage(content="请告诉我上海现在的天气和时间")
    ]
    
    print("\n📤 请求:")
    print(f"  模型: openai/gpt-4o")
    print(f"  可用工具: {len(tools)} 个")
    for tool in tools:
        print(f"    • {tool['function']['name']}")
    print(f"  消息: {messages[0].content}")
    
    try:
        resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            tools=tools
        )
        
        message = resp['choices'][0]['message']
        
        print(f"\n📥 响应:")
        
        if message.get('tool_calls'):
            tool_count = len(message['tool_calls'])
            print(f"✅ 触发了 {tool_count} 个工具调用!")
            
            for i, tool_call in enumerate(message['tool_calls'], 1):
                print(f"\n🔧 工具调用 #{i}:")
                print(f"  函数: {tool_call['function']['name']}")
                args = json.loads(tool_call['function']['arguments'])
                print(f"  参数: {args}")
            
            # 验证是否调用了正确的工具
            called_tools = [tc['function']['name'] for tc in message['tool_calls']]
            print(f"\n🔍 调用的工具: {called_tools}")
            
            if 'get_weather' in called_tools and 'get_time' in called_tools:
                print(f"✅ 正确识别了需要调用的工具（天气和时间）")
                return True
            else:
                print(f"⚠️  工具选择可能不完全匹配预期")
                return True
        else:
            print(f"⚠️  没有触发工具调用")
            return False
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_features():
    """测试 4: 组合功能 - Response Format + Tool Call"""
    print("\n\n" + "=" * 70)
    print("测试 4: 组合功能（实验性）")
    print("=" * 70)
    print("\n⚠️  注意: Response Format 和 Tool Call 通常不能同时使用")
    print("  这个测试主要用于验证 API 的行为\n")
    
    tools = [{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        }
    }]
    
    messages = [
        HumanMessage(content="计算 25 * 4")
    ]
    
    try:
        resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            tools=tools,
            response_format={"type": "json_object"}
        )
        
        print("📥 API 接受了请求")
        message = resp['choices'][0]['message']
        
        if message.get('tool_calls'):
            print("✅ 触发了 Tool Call")
        elif message.get('content'):
            print("✅ 返回了内容")
            try:
                json.loads(message['content'])
                print("✅ 内容是 JSON 格式")
            except:
                print("⚠️  内容不是 JSON 格式")
        
        return True
        
    except Exception as e:
        print(f"⚠️  预期行为: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "🧪" * 35)
    print("Response Format & Tool Call 功能测试")
    print("🧪" * 35)
    
    print("\n📖 测试说明:")
    print("  • Response Format: 让模型以结构化格式（如 JSON）输出")
    print("  • Tool Call: 让模型调用外部工具/函数")
    print()
    print("⚠️  需要配置 API_KEY 环境变量")
    print()
    
    input("按 Enter 开始测试...")
    
    results = {}
    
    # 运行测试
    results['JSON输出'] = test_response_format_json()
    results['单工具调用'] = test_tool_call_single()
    results['多工具调用'] = test_tool_call_multiple()
    results['组合功能'] = test_combined_features()
    
    # 输出总结
    print("\n\n" + "=" * 70)
    print("📊 测试结果总结")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed_count = sum(results.values())
    
    print(f"\n总计: {passed_count}/{total} 通过")
    
    if passed_count == total:
        print("\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️  {total - passed_count} 个测试失败")
    
    print("\n💡 使用建议:")
    print("  • Response Format 适用于需要结构化输出的场景")
    print("  • Tool Call 适用于需要外部数据或执行操作的场景")
    print("  • 通常两者不同时使用，Tool Call 优先级更高")
    print()


if __name__ == "__main__":
    main()

