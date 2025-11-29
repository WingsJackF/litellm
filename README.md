# LiteLLM 简化版模型管理器

一个轻量级的 LLM 模型管理工具，**零依赖 LangChain**，基于 OpenAI SDK，支持多种模型提供商和统一的调用接口。

## ✨ 核心特性

- 🎯 **智能 API 选择**: 根据调用函数自动选择正确的 API 端点
- 🔌 **多提供商支持**: OpenAI, Anthropic, Google, DeepSeek 等（OpenAI 兼容 API）
- 📦 **原始 JSON 响应**: 直接返回完整的 API 响应数据
- 💾 **模型管理**: 自动加载和持久化模型配置
- 🔄 **流式输出**: 完整支持流式响应
- 🛠️ **工具调用**: 支持 Function Calling
- 📊 **详细统计**: 完整的 token 使用和元数据
- 🪶 **轻量级**: 基于 OpenAI 官方 SDK，稳定可靠

## 🚀 快速开始

### 安装依赖

```bash
pip install openai python-dotenv
# 或者
pip install -r requirements.txt
```

### 配置环境

创建 `.env` 文件：

```bash
# 统一 API Key（推荐）
API_KEY=your-api-key-here
BASE_URL=https://api.openai.com/v1
```

### 基础使用

```python
from message_manager import HumanMessage
from model_manager import completion

# 调用模型
messages = [HumanMessage(content="Hello!")]
response = completion(model="openai/gpt-4o", messages=messages)

# 获取结果
print(response['choices'][0]['message']['content'])
print(response['usage'])
```

## 📖 两种 API 调用方式

### completion() - 标准 API（推荐）

用于大多数模型，**自动使用 `chat/completions` 端点**：

```python
from model_manager import completion

# 自动使用 chat/completions API
response = completion(
    model="openai/gpt-4o",
    messages=messages
)

# 返回原始 JSON
# {
#   "id": "chatcmpl-xxx",
#   "model": "gpt-4o",
#   "choices": [...],
#   "usage": {...}
# }
```

**支持的模型：**
- OpenAI: gpt-4o, gpt-4-turbo
- Anthropic: claude-3-5-sonnet-20241022
- Google: gemini-1.5-pro
- DeepSeek: deepseek-chat
- 其他兼容 OpenAI API 的模型

### response() - 新版 API

用于支持 responses API 的模型，**自动使用 `responses` 端点**：

```python
from model_manager import response

# 自动使用 responses API
resp = response(
    model="openai/gpt-5",
    messages=messages
)
```

## 🎯 智能 API 选择

### 关键改进

**不再需要在配置文件中设置 `use_responses_api`！**

调用函数自动决定使用哪个 API：

```python
# ✅ 自动使用 chat/completions
completion(model="openai/gpt-4o", messages=messages)
#   → POST /chat/completions

# ✅ 自动使用 responses
response(model="openai/gpt-5", messages=messages)
#   → POST /responses
```

### 工作原理

```python
# completion() 内部自动设置
model_manager.chat(..., use_responses_api=False)

# response() 内部自动设置
model_manager.chat(..., use_responses_api=True)
```

## 💬 消息格式

使用内置的 Message 类（无需 LangChain）：

```python
from message_manager import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="你是助手"),
    HumanMessage(content="你好"),
    AIMessage(content="你好！"),
    HumanMessage(content="介绍一下你自己")
]

resp = completion(model="openai/gpt-4o", messages=messages)
```

## 📊 响应格式

返回完整的 OpenAI API JSON 格式：

```json
{
  "id": "chatcmpl-xxx",
  "created": 1751494488,
  "model": "gpt-4o",
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Hello! How can I help you?",
        "role": "assistant",
        "tool_calls": null
      }
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 39,
    "total_tokens": 52,
    "prompt_tokens_details": {
      "cached_tokens": 0
    }
  }
}
```

### 提取信息

```python
# 获取内容
content = resp['choices'][0]['message']['content']

# 获取 Token 统计
usage = resp['usage']
print(f"输入: {usage['prompt_tokens']}")
print(f"输出: {usage['completion_tokens']}")
print(f"总计: {usage['total_tokens']}")

# 获取元数据
model_used = resp['model']
response_id = resp['id']
created_at = resp['created']
```

## 🔧 模型名称格式

支持三种格式：

```python
# 1. provider/model（推荐，更清晰）
completion(model="openai/gpt-4o", messages=messages)
completion(model="anthropic/claude-3-5-sonnet-20241022", messages=messages)

# 2. 仅模型名
completion(model="gpt-4o", messages=messages)

# 3. 使用别名（如果在 model.json 中配置）
completion(model="gemini", messages=messages)  # → gemini-pro
```

## 🛠️ 高级功能

### 1. Response Format - JSON 输出

强制模型以 JSON 格式输出：

```python
from message_manager import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是助手，请以 JSON 格式回复"),
    HumanMessage(content="介绍 Python，包含：name, year, features")
]

resp = completion(
    model="openai/gpt-4o",
    messages=messages,
    response_format={"type": "json_object"}
)

# 解析 JSON
import json
content = resp['choices'][0]['message']['content']
data = json.loads(content)
print(data)
# 输出: {"name": "Python", "year": 1991, "features": [...]}
```

### 2. Tool Call - 工具调用

让模型调用外部工具：

```python
from message_manager import HumanMessage

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取城市天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
}]

resp = completion(
    model="openai/gpt-4o",
    messages=[HumanMessage(content="北京天气如何？")],
    tools=tools
)

# 检查工具调用
import json
message = resp['choices'][0]['message']
if message.get('tool_calls'):
    for tool_call in message['tool_calls']:
        func_name = tool_call['function']['name']
        func_args = json.loads(tool_call['function']['arguments'])
        print(f"调用: {func_name}({func_args})")
        # 输出: 调用: get_weather({'city': '北京'})
```

### 3. 视觉模型 - 图片理解

处理包含图片的输入：

```python
from message_manager import HumanMessage

messages = [
    HumanMessage(content=[
        {"type": "text", "text": "这张图片里有什么？"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
    ])
]

resp = completion(model="openai/gpt-4o", messages=messages)
content = resp['choices'][0]['message']['content']
print(content)
```

### 4. 多工具选择

模型可以根据问题选择调用多个工具：

```python
from message_manager import HumanMessage

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取时间",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }
]

resp = completion(
    model="openai/gpt-4o",
    messages=[HumanMessage(content="告诉我北京的天气和时间")],
    tools=tools
)

# 模型可能会调用多个工具
tool_calls = resp['choices'][0]['message'].get('tool_calls', [])
print(f"调用了 {len(tool_calls)} 个工具")
```

## 📋 模型配置

配置保存在 `model.json` 文件中：

```json
{
  "custom_models": [
    {
      "model_name": "my-model",
      "provider": "openai",
      "api_base": "https://api.example.com/v1",
      "max_tokens": 4096
    }
  ],
  "aliases": {
    "gemini": "gemini-pro"
  }
}
```

**注意**: 不再需要设置 `use_responses_api` 字段！

## 📚 示例和测试文件

### 示例文件

- **example_api_calls.py** - API 调用示例（含菜单选择）
- **test_features.py** - 功能测试（Response Format + Tool Call）

### 运行测试

```bash
# 运行交互式示例（可选择测试项）
python example_api_calls.py

# 运行功能测试（自动运行所有测试）
python test_features.py
```

### 测试内容

**test_features.py** 包含：
1. ✅ Response Format - JSON 输出测试
2. ✅ Tool Call - 单个工具调用
3. ✅ Tool Call - 多个工具选择
4. ✅ 组合功能测试

## 🔄 API 对比

| 特性 | completion() | response() |
|------|-------------|-----------|
| **用途** | 大多数模型 | GPT-5 等新模型 |
| **API 端点** | `/chat/completions` | `/responses` |
| **自动设置** | ✅ use_responses_api=False | ✅ use_responses_api=True |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 💡 使用建议

1. **默认使用 `completion()`**：适用于 99% 的场景
2. **仅在需要时使用 `response()`**：如 GPT-5 等明确使用 responses API 的模型
3. **使用 `provider/model` 格式**：更清晰易读
4. **配置环境变量**：使用 `.env` 文件管理敏感信息

## 🐛 常见问题

**Q: 如何选择 completion 还是 response？**  
A: 根据模型的 API 端点：
- GPT-4, Claude, Gemini → `completion()`
- GPT-5 → `response()`

**Q: 还需要在配置文件中设置 use_responses_api 吗？**  
A: **不需要！** 现在调用函数会自动设置。

**Q: 模型名称必须带 provider 前缀吗？**  
A: 不是必须的，但推荐使用 `provider/model` 格式。

**Q: 如何查看完整的 API 请求？**  
A: 响应中包含了所有信息，包括使用的模型、token 统计等。

## 🎉 核心优势

### 之前

```python
# 需要在 model.json 中配置
{
  "model_name": "gpt-5",
  "use_responses_api": true  # 必须手动设置
}

# 调用时还要记住配置
completion(model="gpt-5", messages=messages)  # 可能出错
```

### 现在

```python
# 不需要配置，直接调用正确的函数
response(model="openai/gpt-5", messages=messages)  # ✅ 自动使用正确 API

completion(model="openai/gpt-4o", messages=messages)  # ✅ 自动使用正确 API
```

## 📞 获取帮助

- 查看示例：`example_complete.py`
- 查看配置：`model.json`

---

**Happy Coding! 🎉**
