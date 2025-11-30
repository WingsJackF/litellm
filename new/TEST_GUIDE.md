# 模型管理器测试指南

本文档详细说明 `model_manager.py` 中各测试用例的原理和使用方法。

---

## 目录

1. [测试架构概述](#测试架构概述)
2. [核心函数说明](#核心函数说明)
3. [测试用例详解](#测试用例详解)
   - [OpenAI 模型测试](#1-openai-模型测试)
   - [本地图片上传测试](#2-本地图片上传测试)
   - [Qwen 模型测试](#3-qwen-模型测试)
   - [DeepSeek 模型测试](#4-deepseek-模型测试)
   - [Claude 模型测试](#5-claude-模型测试)
   - [Gemini 模型测试](#6-gemini-模型测试)
   - [Computer Use 测试](#7-computer-use-测试)
4. [环境配置](#环境配置)
5. [运行测试](#运行测试)

---

## 测试架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                      测试框架                                │
├─────────────────────────────────────────────────────────────┤
│  completion()  ──→  chat/completions API  ──→  统一代理     │
│  response()    ──→  responses API         ──→  厂商原始 API │
├─────────────────────────────────────────────────────────────┤
│                    HumanMessage                              │
│                         ↓                                    │
│                  MessageManager                              │
│                    (消息转换)                                 │
│                         ↓                                    │
│              ┌─────────┴─────────┐                          │
│              ↓                   ↓                          │
│       chat/completions      responses                       │
│          (标准格式)         (OpenAI 新格式)                   │
└─────────────────────────────────────────────────────────────┘
```

### API 选择逻辑

| 函数 | API 端点 | 适用场景 |
|------|----------|----------|
| `completion()` | `/chat/completions` | 大多数模型（GPT-4, Claude, Gemini, Qwen, DeepSeek） |
| `response()` | `/responses` | OpenAI 新模型（GPT-5, o3, Computer Use, Deep Research） |

### API Key 优先级

| 参数 | 优先级顺序 |
|------|-----------|
| `use_provider_api=False` (默认) | `API_KEY` → `PROVIDER_API_KEY` |
| `use_provider_api=True` | `PROVIDER_API_KEY` → `API_KEY` |

---

## 核心函数说明

### 1. `completion()` - 标准补全接口

```python
def completion(
    model: str,                    # 模型名称，如 "openai/gpt-4o"
    messages: List[Any],           # 消息列表
    tools: Optional[List[Dict]],   # 工具定义
    response_format: Optional[Dict], # 响应格式
    stream: bool = False,          # 是否流式输出
    response_type: str = "raw",    # "raw" 或 "content"
    use_provider_api: bool = False # 是否使用厂商原始 API
) -> Union[str, Dict]
```

**原理：**
- 使用 OpenAI SDK 调用 `/chat/completions` 端点
- 通过代理服务统一访问多个厂商
- 自动处理消息格式转换

### 2. `response()` - 新版响应接口

```python
def response(
    model: str,                    # 模型名称，如 "openai/gpt-5"
    messages: List[Any],           # 消息列表
    tools: Optional[List[Dict]],   # 工具定义（Computer Use, Deep Research）
    stream: bool = False,          # 是否流式输出
    response_type: str = "raw",    # "raw" 或 "content"
    use_provider_api: bool = False # 是否使用厂商原始 API
) -> Union[str, Dict]
```

**原理：**
- 使用 OpenAI SDK 调用 `/responses` 端点
- 支持新特性：Computer Use、Deep Research
- 消息格式使用 `input` 而非 `messages`

### 3. `HumanMessage` - 消息构造

```python
# 纯文本消息
HumanMessage(content="你好")

# 带图片的消息（支持 URL 和本地路径）
HumanMessage(content=[
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
])

# 本地图片（自动转 base64）
HumanMessage(content=[
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "./local_image.png"}}
])
```

---

## 测试用例详解

### 1. OpenAI 模型测试

#### 1.1 基本问答测试

```python
# 测试 completion API
resp = completion(model="openai/gpt-4o", messages=simple_messages, response_type="content")

# 测试 response API
resp = response(model="openai/gpt-4o", messages=simple_messages, response_type="content")
```

**测试原理：**
- 验证基本的问答功能
- 对比 `completion()` 和 `response()` 两种调用方式
- 验证 `response_type="content"` 只返回文本内容

#### 1.2 结构化输出测试

```python
structured_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "person_info",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "hobbies": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age", "hobbies"]
        }
    }
}

resp = completion(
    model="openai/gpt-4o",
    messages=format_messages,
    response_format=structured_format
)
```

**测试原理：**
- 使用 JSON Schema 约束模型输出格式
- `strict: True` 确保严格遵循 schema
- 验证模型能够生成符合规范的 JSON

#### 1.3 图片理解测试

```python
image_messages = [
    HumanMessage(content=[
        {"type": "text", "text": "这张图片里有什么？"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ])
]

resp = response(model="openai/gpt-5", messages=image_messages)
```

**测试原理：**
- 验证多模态（Vision）能力
- `MessageManager` 自动处理图片格式转换
- 网络 URL 直接传递，本地文件转 base64

#### 1.4 Deep Research 流式测试

```python
deep_research_tools = [
    {"type": "web_search_preview"}  # 必须配置搜索工具
]

stream_resp = response(
    model="openai/o3-deep-research",
    messages=research_messages,
    tools=deep_research_tools,
    stream=True,
    timeout=600  # 长时间任务需要更长超时
)

# 处理流式响应
for event in stream_resp:
    if hasattr(event, 'type'):
        if event.type == 'response.output_text.delta':
            print(event.delta, end="", flush=True)
```

**测试原理：**
- Deep Research 模型需要配置工具（`web_search_preview`、`mcp` 或 `file_search`）
- 使用流式输出避免长时间等待
- 响应事件类型：`response.output_text.delta`（增量文本）、`response.completed`（完成）


**测试原理：**
- Computer Use 需要配置屏幕尺寸和环境
- 返回的动作类型：`click`、`type`、`scroll`、`screenshot` 等
- 坐标基于配置的 `display_width` × `display_height`

---

### 2. 本地图片上传测试

```python
local_image_msgs = [
    HumanMessage(content=[
        {"type": "text", "text": "这张图片里有什么？"},
        {"type": "image_url", "image_url": {"url": "./test_image/img1.webp"}}
    ])
]

resp = completion(model="openai/gpt-4o", messages=local_image_msgs)
```

**测试原理：**
- `MessageManager` 检测到本地文件路径
- 自动读取文件并转换为 base64
- 生成 `data:image/webp;base64,xxxxx` 格式的 URL
- 支持 PNG、JPEG、WebP、GIF 等格式

**转换流程：**
```
本地路径 → 读取文件 → base64 编码 → data URL → API 请求
```

---

### 3. Qwen 模型测试

```python
# 基本问答
resp = completion(model="qwen-plus", messages=simple_messages)

# 结构化输出
resp = completion(
    model="qwen-plus",
    messages=qwen_format_messages,
    response_format=structured_format
)

# 图片理解（Vision 模型）
resp = completion(model="qwen3-vl-plus", messages=image_messages)
```

**测试原理：**
- Qwen 通过代理服务调用，使用 `chat/completions` 接口
- `qwen-plus` 支持文本和结构化输出
- `qwen3-vl-plus` 是视觉模型，支持图片理解

---

### 4. DeepSeek 模型测试

```python
resp = completion(model="deepseek-v3.2-exp", messages=simple_messages)

resp = completion(
    model="deepseek-v3.2-exp",
    messages=deepseek_format_messages,
    response_format=structured_format
)
```

**测试原理：**
- DeepSeek 使用标准 `chat/completions` 接口
- 支持结构化输出（JSON Schema）
- 注意：DeepSeek 不支持 Vision 功能

---

### 5. Claude 模型测试

```python
resp = completion(model="claude-sonnet-4-5-20250929", messages=simple_messages)

resp = completion(
    model="claude-sonnet-4-5-20250929",
    messages=claude_format_messages,
    response_format=structured_format
)
```

**测试原理：**
- Claude 通过代理转换为 OpenAI 兼容格式
- 支持结构化输出
- 注意：Claude 原生不支持 `response_format`，需代理转换

---

### 6. Gemini 模型测试

```python
resp = completion(model="gemini-2.5-pro", messages=simple_messages)

# 图片理解
resp = completion(model="gemini-2.5-pro", messages=image_messages)
```

**测试原理：**
- Google 模型的图片需要转换为 base64 格式
- `MessageManager._is_google_model()` 检测 Google 模型
- 自动将网络 URL 下载并转换为 base64

---

### 7. Computer Use 测试

#### 7.1 OpenAI Computer Use

```python
computer_tool = {
    "type": "computer_use_preview",
    "display_width": 1024,
    "display_height": 768,
    "environment": "mac"
}

computer_messages = [
    HumanMessage(content=[
        {"type": "text", "text": "请点击 model.json 文件"},
        {"type": "image_url", "image_url": {"url": "./screenshot.png"}}
    ])
]

resp = response(
    model="openai/computer-use-preview",
    messages=computer_messages,
    tools=[computer_tool],
    use_provider_api=True,
    truncation="auto"
)

# 解析点击坐标
for item in resp['output']:
    if item.get('type') == 'computer_call':
        action = item.get('action', {})
        if action.get('type') == 'click':
            x, y = action.get('x'), action.get('y')
            print(f"点击位置: ({x}, {y})")
```

**测试原理：**
- 发送截图给模型，模型返回操作指令
- 动作类型：`click`、`type`、`scroll`、`screenshot`、`drag` 等
- 坐标基于 `display_width` × `display_height`
- 需要 `truncation="auto"` 参数

#### 7.2 Anthropic Computer Use

```python
computer_tool = {
    "type": "computer_20250124",
    "name": "computer",
    "display_width_px": 1024,
    "display_height_px": 768,
    "display_number": 1
}

resp = completion(
    model="anthropic/claude-sonnet-4-5-20250514",
    messages=computer_messages,
    tools=[computer_tool],
    use_provider_api=True
)

# Anthropic 返回格式
# action: "left_click", coordinate: [x, y]
```

**测试原理：**
- Anthropic 使用 `beta.messages` API
- 需要 `betas=["computer-use-2025-01-24"]`
- 动作格式：`left_click`、`right_click`、`type`、`screenshot` 等
- 坐标使用 `coordinate: [x, y]` 数组格式

---

## 环境配置

### `.env` 文件配置

```bash
# 统一代理 API（默认使用）
API_KEY=your-proxy-api-key
BASE_URL=https://your-proxy.com/v1

# 厂商原始 API（use_provider_api=True 时使用）
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1

ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_API_BASE=https://api.anthropic.com

GOOGLE_API_KEY=xxx
DEEPSEEK_API_KEY=xxx
```

### 测试图片准备

```bash
# 创建测试目录
mkdir -p test_image

# 截取屏幕并缩放到 1024x768（macOS）
screencapture -x test_image/screenshot.png
sips -z 768 1024 test_image/screenshot.png

# 准备测试图片
cp /path/to/image.jpg test_image/img1.webp
```

---

## 运行测试

```bash
# 安装依赖
pip install -r requirements.txt

# 运行所有测试
python model_manager.py

# 测试结果保存到
# → test_results.md
```

### 测试输出示例

```
🚀 模型管理器测试

==================================================
1️⃣ OpenAI 模型测试
==================================================

📝 基本问答测试completions (gpt-4o)...
🔄 Calling API: https://api.agicto.cn/v1/chat/completions
   Model: gpt-4o, Timeout: 120s
   Response: Hello!

📋 结构化输出测试 (gpt-4o)...
   Structured: 姓名=张三, 年龄=28, 爱好=['阅读', '游泳', '编程']

🖼️ 图片理解测试-网络URL (gpt-5)...
   图片描述: 这是一张风景照片...

==================================================
6. Computer Use 测试 (点击位置测试)
==================================================

找到测试截图: ./test_image/screenshot.png
   图片尺寸: 1024x768

Computer Use 点击测试 (computer-use-preview)...
   模型返回点击动作:
      位置: (125, 320)
      按钮: left
```

---

## 常见问题

### Q1: 为什么调用 Anthropic 模型使用了 OpenAI 的 URL？

**原因：** 模型配置中没有正确解析 provider。

**解决：** 使用 `anthropic/model-name` 格式指定 provider：
```python
completion(model="anthropic/claude-sonnet-4-5", ...)
```

### Q2: Computer Use 返回 404 错误？

**原因：** 模型名称错误或未使用厂商原始 API。

**解决：**
```python
response(
    model="openai/computer-use-preview",
    use_provider_api=True,  # 必须
    truncation="auto"       # 必须
)
```

### Q3: 本地图片无法识别？

**原因：** 文件路径错误或格式不支持。

**解决：**
- 确保文件存在
- 使用支持的格式：PNG、JPEG、WebP、GIF
- 检查路径是否正确

---

## 附录：响应格式对比

### chat/completions 响应

```json
{
  "id": "chatcmpl-xxx",
  "model": "gpt-4o",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello!"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

### responses API 响应

```json
{
  "id": "resp-xxx",
  "model": "gpt-5",
  "output": [
    {
      "type": "message",
      "content": [
        {
          "type": "output_text",
          "text": "Hello!"
        }
      ]
    }
  ]
}
```

### Computer Use 响应

```json
{
  "output": [
    {
      "type": "computer_call",
      "action": {
        "type": "click",
        "x": 125,
        "y": 320,
        "button": "left"
      }
    }
  ]
}
```

