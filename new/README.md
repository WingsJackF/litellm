# 🚀 LLM 统一管理系统

一个简单、强大的 LLM（大语言模型）管理系统，提供统一的模型管理、对话历史管理和多模态支持。

## ✨ 核心特性

### 🎯 模型管理器 (ModelManager)

- **🔄 自动识别** - 支持 15+ 主流模型提供商，自动识别模型配置
- **💾 持久化存储** - 自定义模型自动保存到 JSON，重启后自动加载
- **🌐 统一配置** - 支持 API 代理，统一配置 BASE_URL 和 API_KEY
- **🏷️ 别名支持** - 为模型添加简短易记的别名
- **🔧 完整 CRUD** - 注册、查询、更新、删除模型配置

### 💬 消息管理器 (MessageManager)

- **📝 对话历史** - 完整的对话历史管理和持久化
- **🎨 多模态支持** - 支持文本、图像、音频等多种内容
- **🔢 Token 估算** - 自动估算消息的 Token 数量
- **📊 统计分析** - 消息统计、角色分布、历史导出
- **✅ 消息验证** - 自动验证消息格式和角色交替

### 🤖 聊天演示 (ChatDemo)

- **💡 交互式对话** - 支持实时流式输出
- **🔀 模型切换** - 对话中随时切换模型
- **⚖️ 模型对比** - 同时向多个模型提问，对比回答
- **📦 开箱即用** - 预配置多个主流模型

## 🎁 支持的模型

### 预定义模型（开箱即用）

| 提供商 | 模型 |
|--------|------|
| **OpenAI** | gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo |
| **Anthropic** | claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet, claude-opus-4-5 |
| **DeepSeek** | deepseek-chat, deepseek-coder |
| **Groq** | llama-3.1-8b, llama-3.1-70b, mixtral-8x7b |
| **Mistral** | mistral-small, mistral-medium, mistral-large |

### 自定义模型（任意扩展）

支持注册任意兼容 OpenAI API 格式的模型：
- Google Gemini
- 本地 Ollama 模型
- 私有部署模型
- 其他第三方 API

## 📦 安装

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install python-dotenv openai requests
```

## ⚙️ 配置

### 1. 创建 .env 文件

在项目根目录创建 `.env` 文件：

```bash
# API 代理配置（所有模型使用统一配置）
BASE_URL=https://your-proxy-api.com/v1
API_KEY=your-api-key-here

# 或者为不同提供商设置独立配置
# OPENAI_API_KEY=sk-xxx
# ANTHROPIC_API_KEY=sk-ant-xxx
# DEEPSEEK_API_KEY=sk-xxx
```

### 2. 配置说明

- `BASE_URL`: API 代理地址（统一配置，优先级最高）
- `API_KEY`: API 密钥（统一配置，优先级最高）
- 如果使用官方 API，可以为每个提供商单独配置

## 🚀 快速开始

### 方式 1：交互式聊天（推荐）

```bash
python chat_demo.py
```

选择 5（交互式聊天），即可开始对话！

### 方式 2：代码集成

```python
from model_manager import model_manager
from message_manager import MessageManager
from chat_demo import ChatBot

# 创建聊天机器人
bot = ChatBot(
    model="gpt-4o-mini",
    system_prompt="你是一个有帮助的 AI 助手。",
    method="openai"
)

# 开始对话
response = bot.chat("你好！")
print(response)

# 查看对话历史
bot.print_history()
```

## 📚 详细使用

### 1️⃣ 模型管理

#### 注册自定义模型

```python
from model_manager import model_manager

# 注册本地 Ollama 模型
model_manager.register_model(
    model_name="llama-3-local",
    provider="ollama",
    api_base="http://localhost:11434/v1"
)

# 注册 Google Gemini
model_manager.register_model(
    model_name="gemini-pro",
    provider="google",
    supports_vision=True,
    max_tokens=32768
)

# 添加别名
model_manager.add_model_alias("llama3", "llama-3-local")
```

#### 查询模型信息

```python
# 获取模型配置
model_name, provider, api_key, api_base = model_manager.get_llm_provider("gpt-4")
print(f"模型: {model_name}, 提供商: {provider}")

# 查看详细信息
model_manager.print_model_info("gpt-4")

# 列出所有模型
models = model_manager.list_models()
```

#### 更新和删除

```python
# 更新模型配置
model_manager.update_model(
    "gemini-pro",
    max_tokens=64000,
    supports_functions=True
)

# 删除自定义模型
model_manager.remove_model("my-custom-model")
```

### 2️⃣ 消息管理

```python
from message_manager import MessageManager

# 创建消息管理器
manager = MessageManager(
    system_prompt="你是一个专业的编程助手。",
    max_history=100
)

# 添加消息
manager.add_user_message("如何使用 Python 读取文件？")
manager.add_assistant_message("可以使用 open() 函数...")

# 添加多模态消息
manager.add_multimodal_message(
    role="user",
    text="这张图片是什么？",
    images=["https://example.com/image.jpg"]
)

# 获取消息（用于 API 调用）
messages = manager.get_messages(format="dict")

# 统计信息
print(f"消息数: {len(manager.messages)}")
print(f"Token 估算: {manager.count_tokens_estimate()}")

# 导出历史
manager.export_history("chat_history.json")
```

### 3️⃣ 聊天机器人

```python
from chat_demo import ChatBot

# 创建聊天机器人
bot = ChatBot(
    model="gpt-4o-mini",
    system_prompt="你是一个友好的助手。",
    method="openai"
)

# 普通对话
response = bot.chat("你好！")

# 流式对话（实时输出）
bot.chat_stream("讲一个故事")

# 多模态对话
response = bot.chat(
    "这张图片里有什么？",
    images=["https://example.com/image.jpg"]
)

# 管理对话历史
bot.print_history()      # 查看历史
bot.clear_history()      # 清空历史
bot.export_chat("chat.json")  # 导出历史
```

## 🎮 交互式命令

在交互式聊天模式中，支持以下命令：

```
/help          - 显示帮助信息
/models        - 显示所有可用模型（按提供商分组）
/switch        - 切换模型
/current       - 显示当前模型信息
/stats         - 显示模型统计信息
/history       - 显示对话历史
/clear         - 清空对话历史
/export <file> - 导出对话历史
quit/exit/q    - 退出程序
```

## 🔧 高级功能

### 模型持久化

自定义模型自动保存到 `model.json`，重启后自动加载：

```python
# 第一次运行 - 注册模型
model_manager.register_model("my-model", "openai")

# 重启程序后 - 自动加载
# 模型已经可用，无需重新注册！
bot = ChatBot(model="my-model")
```

### 模型对比

同时向多个模型提问，对比回答：

```bash
python chat_demo.py
# 选择 4（模型对比）
# 输入模型编号：1 5 11
# 输入问题：什么是 Python？
```

### 使用不同的 API 调用方式

```python
# 方式 1: OpenAI SDK（推荐）
bot = ChatBot(model="gpt-4", method="openai")

# 方式 2: requests 库
bot = ChatBot(model="gpt-4", method="requests")

# 方式 3: LiteLLM（需要安装 litellm）
bot = ChatBot(model="gpt-4", method="litellm")
```

## 📁 项目结构

```
new/
├── README.md                    # 项目说明文档
├── model_manager.py             # 模型管理器（支持持久化）
├── message_manager.py           # 消息管理器（支持多模态）
├── chat_demo.py                 # 聊天演示（交互式对话）
├── model.json                   # 模型配置文件（自动生成）
├── 模型持久化说明.md             # 持久化功能详细说明
└── .env                         # 环境变量配置（需自行创建）
```

## 🌟 特色亮点

### 1. 统一管理

- ✅ 一套 API，支持所有主流模型
- ✅ 统一的配置方式（BASE_URL + API_KEY）
- ✅ 自动识别模型提供商

### 2. 开箱即用

- ✅ 预配置 15+ 主流模型
- ✅ 交互式聊天界面
- ✅ 完整的示例代码

### 3. 灵活扩展

- ✅ 轻松添加自定义模型
- ✅ 支持任意 OpenAI 兼容 API
- ✅ 模型配置持久化

### 4. 功能完整

- ✅ 流式输出
- ✅ 多模态支持
- ✅ 对话历史管理
- ✅ Token 统计
- ✅ 模型对比

## 🔍 实际应用场景

### 场景 1：API 代理服务

```python
# .env 配置
BASE_URL=https://your-proxy.com/v1
API_KEY=your-unified-key

# 所有模型都使用代理，无需单独配置
bot = ChatBot(model="gpt-4o-mini")
bot2 = ChatBot(model="claude-3-sonnet")
bot3 = ChatBot(model="deepseek-chat")
# 全部通过代理访问！
```

### 场景 2：本地模型部署

```python
# 注册本地 Ollama 模型
model_manager.register_model(
    model_name="qwen-local",
    provider="ollama",
    api_base="http://localhost:11434/v1"
)

# 像使用云端模型一样使用本地模型
bot = ChatBot(model="qwen-local")
```

### 场景 3：多模型对比

```python
# 同一个问题问不同模型
models = ["gpt-4o-mini", "deepseek-chat", "claude-3-sonnet"]
question = "解释一下量子计算"

for model in models:
    bot = ChatBot(model=model)
    response = bot.chat(question)
    print(f"\n【{model}】\n{response}")
```

### 场景 4：构建客服系统

```python
from chat_demo import ChatBot
from message_manager import MessageManager

# 客服机器人
bot = ChatBot(
    model="gpt-4o-mini",
    system_prompt="你是专业的客服助手，友好、耐心地回答用户问题。"
)

# 处理用户咨询
user_question = "如何退款？"
response = bot.chat(user_question)

# 保存对话记录
bot.export_chat(f"customer_{user_id}_{timestamp}.json")
```

## 📖 详细文档

- 📘 [模型持久化说明](模型持久化说明.md) - 持久化功能详解
- 📗 [快速开始指南](快速开始.md) - 新手入门教程（待创建）
- 📕 [API 文档](API文档.md) - 完整 API 参考（待创建）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发计划

- [ ] 支持更多模型提供商
- [ ] Web UI 界面
- [ ] 流式输出优化
- [ ] Function Calling 支持
- [ ] 对话模板系统
- [ ] 多语言支持

## ❓ 常见问题

### Q1: 如何使用自己的 API 密钥？

A: 在 `.env` 文件中设置：
```bash
API_KEY=your-api-key-here
```

### Q2: 支持哪些模型？

A: 
- **预定义**: OpenAI, Anthropic, DeepSeek, Groq, Mistral 等 15+ 模型
- **自定义**: 支持任何兼容 OpenAI API 格式的模型

### Q3: 如何添加新模型？

A: 
```python
model_manager.register_model("model-name", "provider")
```
模型会自动保存，重启后依然可用。

### Q4: 流式输出有问题？

A: 确保你的 API 支持流式输出，并使用：
```python
bot.chat_stream("你的问题")
```

### Q5: 如何切换模型？

A: 
- 交互模式: 输入 `/switch`
- 代码模式: 创建新的 `ChatBot(model="new-model")`

### Q6: 对话历史在哪里？

A: 在 `MessageManager` 中管理，可以导出为 JSON：
```python
bot.export_chat("chat.json")
```

## 📝 更新日志

### v1.0.0 (2024)

- ✅ 模型管理器（支持 15+ 提供商）
- ✅ 消息管理器（支持多模态）
- ✅ 交互式聊天
- ✅ 模型持久化
- ✅ 流式输出
- ✅ 模型对比
- ✅ 完整文档

## 📄 许可证

MIT License

## 🙏 致谢

本项目基于以下优秀项目的设计思路：

- [LiteLLM](https://github.com/BerriAI/litellm) - 统一的 LLM API
- [OpenAI Python SDK](https://github.com/openai/openai-python) - OpenAI 官方 SDK

## 📧 联系方式

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Email: your-email@example.com

---

**⭐ 如果这个项目对你有帮助，请给一个 Star！**

**🚀 开始使用：`python chat_demo.py`**

