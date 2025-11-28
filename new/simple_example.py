"""
简单示例：使用 completion() 函数调用模型
现在默认返回原始 JSON 格式响应
"""

from langchain_core.messages import HumanMessage, SystemMessage
from model_manager import completion
import json


def example_1_basic():
    """示例 1: 基础调用"""
    print("=" * 60)
    print("示例 1: 基础调用 - 返回原始 JSON")
    print("=" * 60)
    
    try:
        messages = [
            HumanMessage(content="用一句话介绍 Python")
        ]
        
        print("\n📤 调用模型: openai/gpt-4o")
        resp = completion(
            model="openai/gpt-4o",
            messages=messages
        )
        
        # 现在返回的是原始 JSON 格式
        print(f"\n✅ 返回类型: {type(resp).__name__}")  # dict
        print(f"\n📋 完整响应:")
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        
        # 提取信息
        print(f"\n💬 提取信息:")
        print(f"  响应 ID: {resp['id']}")
        print(f"  模型: {resp['model']}")
        print(f"  内容: {resp['choices'][0]['message']['content']}")
        print(f"  输入 tokens: {resp['usage']['prompt_tokens']}")
        print(f"  输出 tokens: {resp['usage']['completion_tokens']}")
        print(f"  总计 tokens: {resp['usage']['total_tokens']}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_2_extract_content():
    """示例 2: 快速提取内容"""
    print("\n" + "=" * 60)
    print("示例 2: 快速提取内容")
    print("=" * 60)
    
    try:
        messages = [
            SystemMessage(content="你是一个有用的助手"),
            HumanMessage(content="Hello!")
        ]
        
        resp = completion(model="openai/gpt-4o", messages=messages)
        
        # 快速访问响应内容
        content = resp['choices'][0]['message']['content']
        usage = resp['usage']
        
        print(f"\n💬 响应内容: {content}")
        print(f"📊 Token 使用: {usage}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def example_3_multi_turn():
    """示例 3: 多轮对话"""
    print("\n" + "=" * 60)
    print("示例 3: 多轮对话")
    print("=" * 60)
    
    try:
        # 构建对话历史
        conversation = [
            SystemMessage(content="你是Python编程助手"),
            HumanMessage(content="什么是列表？")
        ]
        
        # 第一轮
        print("\n👤 用户: 什么是列表？")
        resp1 = completion(model="openai/gpt-4o", messages=conversation)
        assistant_msg = resp1['choices'][0]['message']['content']
        print(f"🤖 助手: {assistant_msg[:100]}...")
        
        # 添加到历史
        from langchain_core.messages import AIMessage
        conversation.append(AIMessage(content=assistant_msg))
        
        # 第二轮
        conversation.append(HumanMessage(content="给个例子"))
        print("\n👤 用户: 给个例子")
        resp2 = completion(model="openai/gpt-4o", messages=conversation)
        assistant_msg2 = resp2['choices'][0]['message']['content']
        print(f"🤖 助手: {assistant_msg2[:100]}...")
        
        print(f"\n📊 总 Token 使用: {resp1['usage']['total_tokens'] + resp2['usage']['total_tokens']}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def example_4_different_models():
    """示例 4: 不同模型调用"""
    print("\n" + "=" * 60)
    print("示例 4: 调用不同模型")
    print("=" * 60)
    
    models = [
        "openai/gpt-4o",
        "gpt-4o",  # 也可以不带 provider 前缀
    ]
    
    messages = [HumanMessage(content="Say hello")]
    
    for model_name in models:
        try:
            print(f"\n📤 测试模型: {model_name}")
            resp = completion(model=model_name, messages=messages)
            
            print(f"  ✅ 成功")
            print(f"  模型: {resp['model']}")
            print(f"  内容: {resp['choices'][0]['message']['content']}")
            print(f"  Tokens: {resp['usage']['total_tokens']}")
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)[:100]}")


def helper_function():
    """辅助函数：简化调用"""
    def chat(user_message: str, model: str = "openai/gpt-4o", system_prompt: str = None):
        """简化的聊天函数"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_message))
        
        resp = completion(model=model, messages=messages)
        return resp['choices'][0]['message']['content']
    
    # 使用
    print("\n" + "=" * 60)
    print("示例 5: 使用辅助函数简化调用")
    print("=" * 60)
    
    try:
        content = chat(
            user_message="用一句话解释机器学习",
            system_prompt="你是AI专家"
        )
        print(f"\n💬 响应: {content}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def main():
    """运行所有示例"""
    print("\n" + "🎯" * 30)
    print("Completion API 简单示例")
    print("现在默认返回原始 JSON 格式")
    print("🎯" * 30)
    
    print("\n⚠️  提示: 需要配置 API_KEY 环境变量")
    print("在 .env 文件中设置: API_KEY=your-key-here\n")
    
    user_input = input("是否运行示例？(y/n): ")
    if user_input.lower() == 'y':
        example_1_basic()
        example_2_extract_content()
        example_3_multi_turn()
        example_4_different_models()
        helper_function()
    
    print("\n" + "=" * 60)
    print("🎉 示例完成!")
    print("=" * 60)
    
    print("\n📚 使用总结:")
    print("```python")
    print("# 调用模型（返回原始 JSON）")
    print("resp = completion(model='openai/gpt-4o', messages=messages)")
    print()
    print("# 提取内容")
    print("content = resp['choices'][0]['message']['content']")
    print("usage = resp['usage']")
    print("model_used = resp['model']")
    print("response_id = resp['id']")
    print("```")
    print()


if __name__ == "__main__":
    main()

