"""
完整示例：如何使用 model_manager 和 message_manager
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model_manager import completion
from message_manager import MessageManager
import json


def example_1_direct_use():
    """示例 1: 直接使用（推荐）- 不需要手动使用 MessageManager"""
    print("=" * 60)
    print("示例 1: 直接使用 completion()")
    print("=" * 60)
    
    try:
        # 1. 创建 LangChain Message 对象
        messages = [
            SystemMessage(content="你是一个有用的助手"),
            HumanMessage(content="用一句话介绍 Python")
        ]
        
        # 2. 直接调用（MessageManager 在内部自动使用）
        print("\n📤 调用模型...")
        resp = completion(model="openai/gpt-4o", messages=messages)
        
        # 3. 使用原始 JSON 响应
        print(f"\n✅ 响应类型: {type(resp).__name__}")  # dict
        print(f"💬 内容: {resp['choices'][0]['message']['content']}")
        print(f"📊 Token 使用: {resp['usage']}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


def example_2_view_conversion():
    """示例 2: 查看 MessageManager 的格式转换（调试用）"""
    print("\n" + "=" * 60)
    print("示例 2: 查看 MessageManager 格式转换")
    print("=" * 60)
    
    # 创建消息
    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮你的？"),
        HumanMessage(content="介绍一下你自己")
    ]
    
    # 方式 1: 转换为 chat/completions 格式
    print("\n1️⃣ Chat/Completions 格式（标准）:")
    msg_manager_chat = MessageManager(api_type="chat/completions")
    api_format_chat = msg_manager_chat(messages)
    print(json.dumps(api_format_chat, indent=2, ensure_ascii=False))
    
    # 方式 2: 转换为 responses 格式
    print("\n2️⃣ Responses 格式（GPT-5）:")
    msg_manager_resp = MessageManager(api_type="responses")
    api_format_resp = msg_manager_resp(messages)
    print(json.dumps(api_format_resp, indent=2, ensure_ascii=False))
    
    print("\n💡 注意: 调用 completion() 时，MessageManager 会自动选择正确的格式")


def example_3_multi_turn_conversation():
    """示例 3: 多轮对话完整流程"""
    print("\n" + "=" * 60)
    print("示例 3: 多轮对话")
    print("=" * 60)
    
    try:
        # 初始化对话历史
        conversation = [
            SystemMessage(content="你是 Python 编程助手")
        ]
        
        # 第一轮对话
        print("\n👤 用户: 什么是列表推导式？")
        conversation.append(HumanMessage(content="什么是列表推导式？"))
        
        resp1 = completion(model="openai/gpt-4o", messages=conversation)
        assistant_reply1 = resp1['choices'][0]['message']['content']
        print(f"🤖 助手: {assistant_reply1[:100]}...")
        
        # 将助手回复添加到历史
        conversation.append(AIMessage(content=assistant_reply1))
        
        # 第二轮对话
        print("\n👤 用户: 给我一个例子")
        conversation.append(HumanMessage(content="给我一个例子"))
        
        resp2 = completion(model="openai/gpt-4o", messages=conversation)
        assistant_reply2 = resp2['choices'][0]['message']['content']
        print(f"🤖 助手: {assistant_reply2[:100]}...")
        
        # 统计
        print(f"\n📊 对话统计:")
        print(f"  消息数量: {len(conversation) + 1}")  # +1 for latest response
        print(f"  第一轮 Token: {resp1['usage']['total_tokens']}")
        print(f"  第二轮 Token: {resp2['usage']['total_tokens']}")
        print(f"  总计 Token: {resp1['usage']['total_tokens'] + resp2['usage']['total_tokens']}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_4_with_vision():
    """示例 4: 视觉模型（带图片）"""
    print("\n" + "=" * 60)
    print("示例 4: 视觉模型消息格式")
    print("=" * 60)
    
    # 创建包含图片的消息
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "这张图片里有什么？"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            }
        ])
    ]
    
    # 查看转换后的格式
    msg_manager = MessageManager(api_type="chat/completions")
    api_format = msg_manager(messages)
    
    print("\n📋 包含图片的消息格式:")
    print(json.dumps(api_format, indent=2, ensure_ascii=False))
    
    print("\n💡 使用方式:")
    print("resp = completion(model='openai/gpt-4o', messages=messages)")
    print("# 内部会自动转换并发送")


def example_5_comparison():
    """示例 5: 对比两种 API 格式"""
    print("\n" + "=" * 60)
    print("示例 5: Chat/Completions vs Responses API")
    print("=" * 60)
    
    messages = [HumanMessage(content="Hello!")]
    
    # Chat/Completions 格式
    msg_manager_chat = MessageManager(api_type="chat/completions")
    chat_format = msg_manager_chat(messages)
    
    # Responses 格式
    msg_manager_resp = MessageManager(api_type="responses")
    resp_format = msg_manager_resp(messages)
    
    print("\n📋 Chat/Completions 格式（GPT-4, Claude, Gemini）:")
    print(json.dumps(chat_format, indent=2))
    
    print("\n📋 Responses 格式（GPT-5）:")
    print(json.dumps(resp_format, indent=2))
    
    print("\n🔑 关键区别:")
    print("  • Chat/Completions: 'type': 'text'")
    print("  • Responses:        'type': 'input_text'")


def main():
    """运行示例"""
    print("\n" + "🎯" * 30)
    print("MessageManager + ModelManager 完整使用指南")
    print("🎯" * 30)
    
    print("\n📖 核心概念:")
    print("  • MessageManager: 消息格式转换工具")
    print("  • ModelManager:   模型调用和管理")
    print("  • completion():   统一调用接口\n")
    
    # 展示格式（不需要 API Key）
    example_2_view_conversion()
    example_4_with_vision()
    example_5_comparison()
    
    # 实际调用（需要 API Key）
    print("\n" + "=" * 60)
    user_input = input("是否运行实际 API 调用示例？(需要 API Key) (y/n): ")
    if user_input.lower() == 'y':
        example_1_direct_use()
        example_3_multi_turn_conversation()
    
    print("\n" + "=" * 60)
    print("🎉 示例完成!")
    print("=" * 60)
    
    print("\n📚 使用总结:")
    print("""
    ┌─────────────────────────────────────────────────────┐
    │ 1. 创建 LangChain Message 对象                      │
    │    messages = [HumanMessage(content="Hello")]      │
    │                                                     │
    │ 2. 调用 completion()                                │
    │    resp = completion(model="openai/gpt-4o",        │
    │                      messages=messages)            │
    │                                                     │
    │ 3. MessageManager 自动在内部使用，无需手动调用      │
    │                                                     │
    │ 4. 获取原始 JSON 响应                               │
    │    content = resp['choices'][0]['message']['content']│
    │    usage = resp['usage']                           │
    └─────────────────────────────────────────────────────┘
    
    💡 通常情况下，你只需要：
       1. 导入: from model_manager import completion
       2. 调用: resp = completion(model, messages)
       3. 使用: resp['choices'][0]['message']['content']
       
    🔧 MessageManager 只在以下情况手动使用：
       • 调试消息格式
       • 查看 API 请求结构
       • 理解格式转换过程
    """)


if __name__ == "__main__":
    main()

