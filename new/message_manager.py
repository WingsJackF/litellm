from typing import List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
import base64
import requests
import os
from urllib.parse import urlparse
from pathlib import Path


def get_mime_type(file_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = Path(file_path).suffix.lower()
    mime_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.svg': 'image/svg+xml',
    }
    return mime_map.get(ext, 'image/jpeg')


def local_image_to_base64(file_path: str) -> Dict[str, str]:
    """
    将本地图片文件转换为 base64 编码
    
    Args:
        file_path: 本地图片文件路径
        
    Returns:
        Dict: {"mimeType": "image/jpeg", "data": "base64字符串"}
    """
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ 图片文件不存在: {file_path}")
            return {"mimeType": "image/jpeg", "data": ""}
        
        # 读取文件并转换为 base64
        with open(path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # 获取 MIME 类型
        mime_type = get_mime_type(file_path)
        
        return {
            "mimeType": mime_type,
            "data": image_data
        }
    except Exception as e:
        print(f"⚠️ 本地图片转换失败: {e}")
        return {"mimeType": "image/jpeg", "data": ""}


def url_to_base64(url: str) -> Dict[str, str]:
    """
    将图片 URL 转换为 base64 编码
    
    Args:
        url: 图片 URL
        
    Returns:
        Dict: {"mimeType": "image/jpeg", "data": "base64字符串"}
    """
    try:
        # 下载图片
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 获取 MIME 类型
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        if ';' in content_type:
            content_type = content_type.split(';')[0].strip()
        
        # 如果无法从 header 获取，尝试从 URL 推断
        if content_type == 'application/octet-stream' or not content_type.startswith('image/'):
            content_type = get_mime_type(urlparse(url).path)
        
        # 转换为 base64
        image_data = base64.b64encode(response.content).decode('utf-8')
        
        return {
            "mimeType": content_type,
            "data": image_data
        }
    except Exception as e:
        print(f"⚠️ 图片 URL 转换失败: {e}")
        return {"mimeType": "image/jpeg", "data": ""}


def image_to_base64(image_source: str) -> Dict[str, str]:
    """
    将图片转换为 base64 编码（自动判断是 URL 还是本地文件）
    
    Args:
        image_source: 图片来源，可以是 URL 或本地文件路径
        
    Returns:
        Dict: {"mimeType": "image/jpeg", "data": "base64字符串"}
    """
    # 判断是 URL 还是本地文件
    if image_source.startswith(('http://', 'https://', 'data:')):
        # 如果已经是 data URL，直接解析
        if image_source.startswith('data:'):
            try:
                # 格式: data:image/png;base64,xxxxx
                header, data = image_source.split(',', 1)
                mime_type = header.split(':')[1].split(';')[0]
                return {"mimeType": mime_type, "data": data}
            except:
                return {"mimeType": "image/jpeg", "data": ""}
        return url_to_base64(image_source)
    else:
        # 本地文件路径
        return local_image_to_base64(image_source)


class Message(BaseModel):
    """消息基类 - 使用 Pydantic BaseModel"""
    content: Union[str, List[Dict[str, Any]]] = Field(
        description="消息内容，可以是字符串或内容块列表"
    )
    role: Literal["user", "assistant", "system"] = Field(
        description="消息角色"
    )
    
    class Config:
        # 允许任意类型（为了兼容性）
        arbitrary_types_allowed = True


class HumanMessage(Message):
    """用户消息"""
    def __init__(self, content: Union[str, List[Dict]], **data):
        super().__init__(content=content, role="user", **data)


class AIMessage(Message):
    """助手消息"""
    def __init__(self, content: Union[str, List[Dict]], **data):
        super().__init__(content=content, role="assistant", **data)


class SystemMessage(Message):
    """系统消息"""
    def __init__(self, content: Union[str, List[Dict]], **data):
        super().__init__(content=content, role="system", **data)


class MessageManager():
    def __init__(self, 
                 api_type: str = "chat/completions",
                 model: str = "o3"):
        self.api_type = api_type
        self.model = model
    
    def _is_google_model(self) -> bool:
        """判断是否是 Google 模型"""
        google_prefixes = ['gemini', 'palm', 'bard']
        model_lower = self.model.lower()
        return any(model_lower.startswith(prefix) for prefix in google_prefixes)
    
    def _get_image_source(self, image_url_data: Union[str, Dict]) -> str:
        """从 image_url 数据中提取图片来源（URL 或本地路径）"""
        if isinstance(image_url_data, dict):
            return image_url_data.get("url", "")
        return image_url_data
    
    def _is_local_file(self, source: str) -> bool:
        """判断是否是本地文件路径"""
        if source.startswith(('http://', 'https://', 'data:')):
            return False
        return os.path.exists(source) or not source.startswith('/')
    
    def _convert_image_to_base64_url(self, image_url_data: Union[str, Dict]) -> Dict:
        """
        将图片转换为 base64 data URL 格式（支持本地文件和网络 URL）
        
        Returns:
            Dict: {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
        """
        source = self._get_image_source(image_url_data)
        
        # 如果已经是 data URL，直接返回
        if source.startswith('data:'):
            return {
                "type": "image_url",
                "image_url": {"url": source}
            }
        
        # 转换为 base64
        base64_data = image_to_base64(source)
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{base64_data['mimeType']};base64,{base64_data['data']}"
            }
        }
        
    def __call__(self, messages: List[Union[HumanMessage, AIMessage]]) -> List[Dict[str, Any]]:
        is_google = self._is_google_model()
        
        if self.api_type == "chat/completions":
            result = []
            for m in messages:
                if isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                elif isinstance(m, SystemMessage):
                    role = "system"
                else:
                    role = "user"  # fallback
                    
                if isinstance(m.content, str):
                    content = [{"type": "text", "text": m.content}]
                elif isinstance(m.content, list):
                    content = []
                    for c in m.content:
                        type_ = c.get("type", "text")
                        if type_ == "text":
                            content.append(c)
                        elif type_ == "image_url":
                            image_source = self._get_image_source(c.get("image_url", {}))
                            is_local = self._is_local_file(image_source)
                            
                            if is_google or is_local:
                                # Google 模型或本地文件：转换为 base64 data URL
                                content.append(self._convert_image_to_base64_url(c.get("image_url", {})))
                            else:
                                # 其他模型 + 网络 URL：保持原样
                                content.append(c)
                        else:
                            content.append(c)
                else:
                    content = m.content
                    
                result.append({"role": role, "content": content})
            return result
        elif self.api_type == "responses":
            result = []
            for m in messages:
                if isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                elif isinstance(m, SystemMessage):
                    role = "system"
                else:
                    role = "user"  # fallback
                    
                if isinstance(m.content, str):
                    content = [{"type": "input_text", "text": m.content}]
                elif isinstance(m.content, list):
                    content = []
                    for c in m.content:
                        type_ = c.get("type", "input_text")
                        if type_ == "text":
                            content.append({
                                "type": "input_text",
                                "text": c.get("text", "")
                            })
                        elif type_ == "image_url":
                            image_source = self._get_image_source(c.get("image_url", {}))
                            is_local = self._is_local_file(image_source)
                            
                            if is_local:
                                # 本地文件：转换为 base64 data URL
                                base64_data = image_to_base64(image_source)
                                url = f"data:{base64_data['mimeType']};base64,{base64_data['data']}"
                            elif isinstance(c.get("image_url", {}), dict):
                                url = c.get("image_url", {}).get("url", "")
                            else:
                                url = c.get("image_url", "")
                            
                            content.append({
                                "type": "input_image",
                                "image_url": url
                            })
                result.append({"role": role, "content": content})
            return result
                
                    