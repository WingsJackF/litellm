"""
快速测试 completion() 和 response() 函数
"""

from langchain_core.messages import HumanMessage, SystemMessage
from model_manager import completion, response


def test_completion():
    """测试 completion API"""
    print("=" * 60)
    print("测试 completion() 函数")
    print("=" * 60)
    
    try:
        messages = [
            SystemMessage(content="你是一个有用的助手，回答简洁明了。"),
            HumanMessage(content="用一句话解释什么是 Python")
        ]
        
        print("\n📤 调用模型: openai/gpt-4o")
        print(f"💬 消息数量: {len(messages)}")
        
        resp = completion(
            model="openai/gpt-4o",
            messages=messages
        )
        
        print(f"\n✅ 成功!")
        print(f"📥 响应: {resp.content}")
        print(f"📊 类型: {type(resp).__name__}")
        
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()


def test_model_formats():
    """测试不同的模型名称格式"""
    print("\n" + "=" * 60)
    print("测试模型名称格式")
    print("=" * 60)
    
    formats = [
        "openai/gpt-4o",      # 带 provider
        "gpt-4o",             # 不带 provider
    ]
    
    messages = [HumanMessage(content="Hi")]
    
    for model_format in formats:
        try:
            print(f"\n📤 测试格式: {model_format}")
            resp = completion(model=model_format, messages=messages)
            print(f"   ✅ 成功: {resp.content[:50]}...")
        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:100]}")


def test_response_api_warning():
    """测试 response API 警告"""
    print("\n" + "=" * 60)
    print("测试 response() 函数警告")
    print("=" * 60)
    
    print("\n💡 response() 用于支持 responses API 的模型（如 gpt-5）")
    print("如果模型未配置 use_responses_api=true，会显示警告\n")
    
    try:
        messages = [HumanMessage(content="Test")]
        
        # 这里用 gpt-4o 测试会显示警告（因为它不是 responses API 模型）
        print("📤 使用 response() 调用 gpt-4o（会显示警告）:")
        resp = response(model="openai/gpt-4o", messages=messages)
        print(f"✅ 调用成功: {resp.content[:50]}...")
        
    except Exception as e:
        print(f"❌ 失败: {e}")


def main():
    """运行测试"""
    print("\n" + "🧪" * 30)
    print("Completion & Response API 快速测试")
    print("🧪" * 30 + "\n")
    
    print("⚠️  提示: 需要配置 API_KEY 环境变量")
    print("在 .env 文件中设置: API_KEY=your-key-here\n")
    
    # 运行测试
    test_completion()
    test_model_formats()
    test_response_api_warning()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("=" * 60)
    
    print("\n📚 更多示例:")
    print("  - example_api_calls.py  完整的 API 调用示例")
    print("  - example_usage.py      MessageManager 使用示例")
    print("  - test_model_manager.py 完整测试套件")
    print()


if __name__ == "__main__":
    main()

