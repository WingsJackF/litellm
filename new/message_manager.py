"""
简化版消息管理器
支持多模态消息管理、对话历史管理和消息验证
"""

import json
from typing import List, Dict, Optional, Union, Literal, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MessageContent:
    """消息内容类（支持多模态）"""
    type: ContentType
    content: Union[str, Dict]
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        if self.type == ContentType.TEXT:
            return {"type": "text", "text": self.content}
        elif self.type == ContentType.IMAGE_URL:
            return {
                "type": "image_url",
                "image_url": {"url": self.content}
            }
        elif self.type == ContentType.IMAGE_BASE64:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.content}"}
            }
        else:
            return {"type": self.type.value, "content": self.content}


@dataclass
class Message:
    """消息类"""
    role: MessageRole
    content: Union[str, List[MessageContent]]
    timestamp: datetime = field(default_factory=datetime.now)
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为标准 OpenAI 格式"""
        msg = {"role": self.role.value}
        
        # 处理内容
        if isinstance(self.content, str):
            msg["content"] = self.content
        elif isinstance(self.content, list):
            msg["content"] = [c.to_dict() for c in self.content]
        
        # 添加可选字段
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        
        return msg
    
    def is_multimodal(self) -> bool:
        """检查是否是多模态消息"""
        return isinstance(self.content, list) and len(self.content) > 0


class MessageManager:
    """
    消息管理器 - 管理对话历史和多模态消息
    
    功能：
    1. 管理对话历史
    2. 支持多模态消息（文本、图像、音频等）
    3. 消息验证
    4. 消息格式转换
    5. 对话历史导出/导入
    """
    
    def __init__(self, system_prompt: Optional[str] = None, max_history: int = 100):
        """
        初始化消息管理器
        
        Args:
            system_prompt: 系统提示词
            max_history: 最大历史消息数量
        """
        self.messages: List[Message] = []
        self.max_history = max_history
        
        # 如果提供了系统提示词，添加为第一条消息
        if system_prompt:
            self.add_system_message(system_prompt)
    
    def add_message(
        self,
        role: Union[MessageRole, str],
        content: Union[str, List[MessageContent], List[Dict]],
        **kwargs
    ) -> Message:
        """
        添加消息
        
        Args:
            role: 消息角色
            content: 消息内容（可以是字符串或多模态内容列表）
            **kwargs: 其他参数（name, tool_calls 等）
            
        Returns:
            Message: 创建的消息对象
        """
        # 转换角色
        if isinstance(role, str):
            role = MessageRole(role)
        
        # 处理多模态内容
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict):
                # 从字典转换
                processed_content = []
                for item in content:
                    if item["type"] == "text":
                        processed_content.append(
                            MessageContent(ContentType.TEXT, item.get("text", ""))
                        )
                    elif item["type"] == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            processed_content.append(
                                MessageContent(ContentType.IMAGE_BASE64, url)
                            )
                        else:
                            processed_content.append(
                                MessageContent(ContentType.IMAGE_URL, url)
                            )
                content = processed_content
        
        # 创建消息
        message = Message(role=role, content=content, **kwargs)
        
        # 验证消息
        self._validate_message(message)
        
        # 添加到历史
        self.messages.append(message)
        
        # 检查历史长度限制
        self._trim_history()
        
        return message
    
    def add_system_message(self, content: str, **kwargs) -> Message:
        """添加系统消息"""
        return self.add_message(MessageRole.SYSTEM, content, **kwargs)
    
    def add_user_message(
        self,
        content: Union[str, List[MessageContent], List[Dict]],
        **kwargs
    ) -> Message:
        """添加用户消息（支持多模态）"""
        return self.add_message(MessageRole.USER, content, **kwargs)
    
    def add_assistant_message(self, content: str, **kwargs) -> Message:
        """添加助手消息"""
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)
    
    def add_text_message(self, role: Union[MessageRole, str], text: str) -> Message:
        """添加纯文本消息"""
        return self.add_message(role, text)
    
    def add_multimodal_message(
        self,
        role: Union[MessageRole, str],
        text: str,
        images: Optional[List[str]] = None,
        **kwargs
    ) -> Message:
        """
        添加多模态消息（文本 + 图像）
        
        Args:
            role: 消息角色
            text: 文本内容
            images: 图像 URL 或 Base64 列表
            **kwargs: 其他参数
        """
        contents = [MessageContent(ContentType.TEXT, text)]
        
        if images:
            for img in images:
                if img.startswith("data:image") or img.startswith("data:image"):
                    contents.append(MessageContent(ContentType.IMAGE_BASE64, img))
                else:
                    contents.append(MessageContent(ContentType.IMAGE_URL, img))
        
        return self.add_message(role, contents, **kwargs)
    
    def _validate_message(self, message: Message):
        """验证消息格式"""
        # 检查角色交替（可选）
        if len(self.messages) > 0:
            last_role = self.messages[-1].role
            current_role = message.role
            
            # 系统消息只能在开头
            if current_role == MessageRole.SYSTEM and len(self.messages) > 1:
                if self.messages[-1].role != MessageRole.SYSTEM:
                    print("⚠️  警告: 系统消息通常应该在对话开始时添加")
        
        # 验证内容不为空
        if isinstance(message.content, str) and not message.content.strip():
            raise ValueError("消息内容不能为空")
        
        return True
    
    def _trim_history(self):
        """修剪历史记录，保持在最大长度内"""
        if len(self.messages) > self.max_history:
            # 保留系统消息（如果存在）
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            other_messages = [m for m in self.messages if m.role != MessageRole.SYSTEM]
            
            # 只保留最近的消息
            keep_count = self.max_history - len(system_messages)
            other_messages = other_messages[-keep_count:]
            
            self.messages = system_messages + other_messages
            print(f"⚠️  历史消息已修剪至 {len(self.messages)} 条")
    
    def get_messages(self, format: Literal["object", "dict"] = "dict") -> List:
        """
        获取所有消息
        
        Args:
            format: 返回格式 ("object" 或 "dict")
            
        Returns:
            消息列表
        """
        if format == "dict":
            return [m.to_dict() for m in self.messages]
        return self.messages
    
    def get_recent_messages(self, count: int = 10, format: Literal["object", "dict"] = "dict") -> List:
        """获取最近的 N 条消息"""
        recent = self.messages[-count:]
        if format == "dict":
            return [m.to_dict() for m in recent]
        return recent
    
    def clear_history(self, keep_system: bool = True):
        """
        清空对话历史
        
        Args:
            keep_system: 是否保留系统消息
        """
        if keep_system:
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            self.messages = system_messages
        else:
            self.messages = []
        print("✅ 对话历史已清空")
    
    def pop_last_message(self) -> Optional[Message]:
        """移除并返回最后一条消息"""
        if self.messages:
            return self.messages.pop()
        return None
    
    def count_messages(self) -> Dict[str, int]:
        """统计各角色的消息数量"""
        counts = {role.value: 0 for role in MessageRole}
        for msg in self.messages:
            counts[msg.role.value] += 1
        return counts
    
    def count_tokens_estimate(self) -> int:
        """
        估算消息的 token 数量（简单估算）
        实际应用中应该使用 tiktoken 等库
        """
        total = 0
        for msg in self.messages:
            if isinstance(msg.content, str):
                # 简单估算：4 个字符约等于 1 个 token
                total += len(msg.content) // 4
            elif isinstance(msg.content, list):
                for content in msg.content:
                    if content.type == ContentType.TEXT:
                        total += len(str(content.content)) // 4
                    else:
                        # 图像等多模态内容按固定 token 计算
                        total += 85  # OpenAI 的图像 token 数
        return total
    
    def export_history(self, filepath: str):
        """导出对话历史到 JSON 文件"""
        data = {
            "exported_at": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content if isinstance(m.content, str) else [c.to_dict() for c in m.content],
                    "timestamp": m.timestamp.isoformat(),
                    "name": m.name,
                    "tool_calls": m.tool_calls,
                    "metadata": m.metadata
                }
                for m in self.messages
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 对话历史已导出到: {filepath}")
    
    def import_history(self, filepath: str):
        """从 JSON 文件导入对话历史"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.messages = []
        for msg_data in data["messages"]:
            role = MessageRole(msg_data["role"])
            content = msg_data["content"]
            
            message = Message(
                role=role,
                content=content,
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                name=msg_data.get("name"),
                tool_calls=msg_data.get("tool_calls"),
                metadata=msg_data.get("metadata", {})
            )
            self.messages.append(message)
        
        print(f"✅ 已导入 {len(self.messages)} 条消息")
    
    def format_for_display(self) -> str:
        """格式化消息用于显示"""
        output = []
        output.append("=" * 60)
        output.append(f"📝 对话历史 (共 {len(self.messages)} 条消息)")
        output.append("=" * 60)
        
        for i, msg in enumerate(self.messages, 1):
            # 角色图标
            role_icons = {
                MessageRole.SYSTEM: "⚙️",
                MessageRole.USER: "👤",
                MessageRole.ASSISTANT: "🤖",
                MessageRole.TOOL: "🔧"
            }
            icon = role_icons.get(msg.role, "💬")
            
            output.append(f"\n{i}. {icon} {msg.role.value.upper()}")
            output.append(f"   时间: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 内容
            if isinstance(msg.content, str):
                # 文本内容
                content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                output.append(f"   内容: {content_preview}")
            elif isinstance(msg.content, list):
                # 多模态内容
                output.append(f"   内容: [多模态消息]")
                for content in msg.content:
                    if content.type == ContentType.TEXT:
                        text = str(content.content)[:50] + "..." if len(str(content.content)) > 50 else content.content
                        output.append(f"     - 文本: {text}")
                    else:
                        output.append(f"     - {content.type.value}")
            
            if msg.is_multimodal():
                output.append(f"   🎨 多模态消息")
        
        output.append("\n" + "=" * 60)
        output.append(f"📊 统计: {self.count_messages()}")
        output.append(f"🔢 估算 Token 数: ~{self.count_tokens_estimate()}")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def print_history(self):
        """打印对话历史"""
        print(self.format_for_display())
    
    def ensure_alternating_roles(self):
        """确保用户和助手消息交替（修复格式问题）"""
        if len(self.messages) < 2:
            return
        
        fixed_messages = []
        last_role = None
        
        for msg in self.messages:
            # 系统消息始终保留
            if msg.role == MessageRole.SYSTEM:
                fixed_messages.append(msg)
                continue
            
            # 如果连续两条相同角色，合并它们
            if last_role == msg.role and len(fixed_messages) > 0:
                last_msg = fixed_messages[-1]
                if isinstance(last_msg.content, str) and isinstance(msg.content, str):
                    last_msg.content += "\n" + msg.content
                    continue
            
            fixed_messages.append(msg)
            last_role = msg.role
        
        if len(fixed_messages) != len(self.messages):
            self.messages = fixed_messages
            print(f"✅ 已修复消息格式，合并了 {len(self.messages) - len(fixed_messages)} 条重复角色消息")


# 便捷函数
def create_text_message(role: str, text: str) -> Dict:
    """创建文本消息（字典格式）"""
    return {"role": role, "content": text}


def create_multimodal_message(role: str, text: str, image_urls: List[str]) -> Dict:
    """创建多模态消息（字典格式）"""
    content = [{"type": "text", "text": text}]
    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })
    return {"role": role, "content": content}


if __name__ == "__main__":
    # 示例用法
    print("🚀 消息管理器示例\n")
    
    # 1. 创建消息管理器
    print("1️⃣ 创建消息管理器:")
    manager = MessageManager(system_prompt="你是一个有帮助的 AI 助手。")
    print(f"   已初始化，当前消息数: {len(manager.messages)}\n")
    
    # 2. 添加文本消息
    print("2️⃣ 添加文本消息:")
    manager.add_user_message("你好！")
    manager.add_assistant_message("你好！我是 AI 助手，有什么可以帮助你的吗？")
    manager.add_user_message("请介绍一下 Python。")
    manager.add_assistant_message("Python 是一种高级编程语言，以其简洁易读的语法而闻名...")
    print(f"   已添加 4 条消息\n")
    
    # 3. 添加多模态消息
    print("3️⃣ 添加多模态消息:")
    manager.add_multimodal_message(
        role="user",
        text="这张图片是什么？",
        images=["https://example.com/image.jpg"]
    )
    print(f"   已添加多模态消息\n")
    
    # 4. 查看对话历史
    print("4️⃣ 查看对话历史:")
    manager.print_history()
    
    # 5. 导出历史
    print("\n5️⃣ 导出对话历史:")
    manager.export_history("/tmp/chat_history.json")
    
    # 6. 统计信息
    print("\n6️⃣ 统计信息:")
    print(f"   消息统计: {manager.count_messages()}")
    print(f"   Token 估算: ~{manager.count_tokens_estimate()}")
    
    # 7. 获取最近的消息（用于 API 调用）
    print("\n7️⃣ 获取 API 格式的消息:")
    api_messages = manager.get_messages(format="dict")
    print(f"   共 {len(api_messages)} 条消息，格式适用于 API 调用")
    print(f"   示例: {api_messages[0]}")

