"""
演示如何获取原始 JSON 格式的 API 响应
"""

from langchain_core.messages import HumanMessage, SystemMessage
from model_manager import completion
import json


def example_default_response():
    """示例 1: 默认返回 AIMessage 对象"""
    print("=" * 60)
    print("示例 1: 默认返回 AIMessage 对象")
    print("=" * 60)
    
    try:
        messages = [
            HumanMessage(content="Say hello in one sentence")
        ]
        
        print("\n📤 调用模型: openai/gpt-4o")
        print("   return_raw=False (默认)")
        
        resp = completion(
            model="openai/gpt-4o",
            messages=messages
        )
        
        print(f"\n📥 返回类型: {type(resp).__name__}")
        print(f"📥 响应内容: {resp.content}")
        print(f"📥 完整对象: {resp}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def example_raw_json_response():
    """示例 2: 返回原始 JSON 格式"""
    print("\n" + "=" * 60)
    print("示例 2: 返回原始 JSON 格式")
    print("=" * 60)
    
    try:
        messages = [
            HumanMessage(content="Say hello in one sentence")
        ]
        
        print("\n📤 调用模型: openai/gpt-4o")
        print("   return_raw=True")
        
        raw_resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            return_raw=True
        )
        
        print(f"\n📥 返回类型: {type(raw_resp).__name__}")
        print(f"\n📥 完整 JSON 响应:")
        print(json.dumps(raw_resp, indent=2, ensure_ascii=False))
        
        # 提取关键信息
        print(f"\n📊 解析响应:")
        print(f"  ID: {raw_resp.get('id')}")
        print(f"  Model: {raw_resp.get('model')}")
        print(f"  Created: {raw_resp.get('created')}")
        print(f"  Content: {raw_resp['choices'][0]['message']['content']}")
        print(f"  Usage: {raw_resp.get('usage')}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_compare_both():
    """示例 3: 对比两种返回格式"""
    print("\n" + "=" * 60)
    print("示例 3: 对比两种返回格式")
    print("=" * 60)
    
    try:
        messages = [
            SystemMessage(content="你是一个助手"),
            HumanMessage(content="介绍一下 Python")
        ]
        
        # AIMessage 格式
        print("\n1️⃣ AIMessage 格式 (return_raw=False):")
        print("-" * 60)
        ai_resp = completion(
            model="openai/gpt-4o",
            messages=messages
        )
        print(f"类型: {type(ai_resp).__name__}")
        print(f"内容: {ai_resp.content[:100]}...")
        
        # JSON 格式
        print("\n2️⃣ JSON 格式 (return_raw=True):")
        print("-" * 60)
        json_resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            return_raw=True
        )
        print(f"类型: {type(json_resp).__name__}")
        print(f"内容: {json_resp['choices'][0]['message']['content'][:100]}...")
        print(f"Token 使用: {json_resp['usage']}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def example_extract_usage():
    """示例 4: 提取 Token 使用统计"""
    print("\n" + "=" * 60)
    print("示例 4: 提取 Token 使用统计")
    print("=" * 60)
    
    try:
        messages = [
            HumanMessage(content="Explain machine learning in 2 sentences")
        ]
        
        raw_resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            return_raw=True
        )
        
        usage = raw_resp.get('usage', {})
        
        print("\n📊 Token 使用统计:")
        print(f"  输入 Tokens: {usage.get('prompt_tokens', 0)}")
        print(f"  输出 Tokens: {usage.get('completion_tokens', 0)}")
        print(f"  总计 Tokens: {usage.get('total_tokens', 0)}")
        
        if 'prompt_tokens_details' in usage:
            details = usage['prompt_tokens_details']
            print(f"\n  缓存 Tokens: {details.get('cached_tokens', 0)}")
        
        print(f"\n💬 响应内容:")
        print(f"  {raw_resp['choices'][0]['message']['content']}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def example_access_metadata():
    """示例 5: 访问完整元数据"""
    print("\n" + "=" * 60)
    print("示例 5: 访问完整元数据")
    print("=" * 60)
    
    try:
        messages = [
            HumanMessage(content="Hello!")
        ]
        
        raw_resp = completion(
            model="openai/gpt-4o",
            messages=messages,
            return_raw=True
        )
        
        print("\n📋 完整元数据:")
        print(f"  Response ID: {raw_resp.get('id')}")
        print(f"  Model: {raw_resp.get('model')}")
        print(f"  Object Type: {raw_resp.get('object')}")
        print(f"  Created At: {raw_resp.get('created')}")
        print(f"  System Fingerprint: {raw_resp.get('system_fingerprint')}")
        
        choice = raw_resp['choices'][0]
        print(f"\n  Choice Index: {choice.get('index')}")
        print(f"  Finish Reason: {choice.get('finish_reason')}")
        
        message = choice['message']
        print(f"\n  Message Role: {message.get('role')}")
        print(f"  Message Content: {message.get('content')[:100]}...")
        print(f"  Tool Calls: {message.get('tool_calls')}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def example_different_providers():
    """示例 6: 不同提供商的原始响应"""
    print("\n" + "=" * 60)
    print("示例 6: 不同提供商的原始响应格式")
    print("=" * 60)
    
    providers = [
        ("openai/gpt-4o", "OpenAI"),
        ("deepseek/deepseek-chat", "DeepSeek"),
    ]
    
    messages = [HumanMessage(content="Hi")]
    
    for model_path, provider_name in providers:
        try:
            print(f"\n📤 {provider_name} ({model_path}):")
            print("-" * 60)
            
            raw_resp = completion(
                model=model_path,
                messages=messages,
                return_raw=True
            )
            
            print(f"  Model: {raw_resp.get('model')}")
            print(f"  Content: {raw_resp['choices'][0]['message']['content'][:80]}...")
            print(f"  Usage: {raw_resp.get('usage')}")
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)[:100]}")


def main():
    """运行所有示例"""
    print("\n" + "🎯" * 30)
    print("原始 JSON 响应格式示例")
    print("🎯" * 30)
    
    print("\n💡 说明:")
    print("  • return_raw=False (默认): 返回 LangChain AIMessage 对象")
    print("  • return_raw=True: 返回原始 OpenAI API JSON 格式")
    print()
    
    print("⚠️  提示: 需要配置 API_KEY 环境变量\n")
    
    user_input = input("是否运行实际 API 调用示例？(y/n): ")
    if user_input.lower() == 'y':
        example_default_response()
        example_raw_json_response()
        example_compare_both()
        example_extract_usage()
        example_access_metadata()
        # example_different_providers()  # 需要多个 provider 的 API key
    
    print("\n" + "=" * 60)
    print("🎉 示例完成!")
    print("=" * 60)
    
    print("\n📚 使用方法总结:")
    print("```python")
    print("# 方式 1: 返回 AIMessage 对象（默认）")
    print("resp = completion(model='openai/gpt-4o', messages=messages)")
    print("print(resp.content)")
    print()
    print("# 方式 2: 返回原始 JSON")
    print("raw_resp = completion(model='openai/gpt-4o', messages=messages, return_raw=True)")
    print("print(raw_resp['choices'][0]['message']['content'])")
    print("print(raw_resp['usage'])")
    print("```")
    print()


if __name__ == "__main__":
    main()

