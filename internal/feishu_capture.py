from __future__ import annotations

import uuid
from typing import Any, Optional


def make_feishu_internal_capture_adapter(config: Any) -> Any:
    from gateway.config import Platform
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    class _FeishuInternalCaptureAdapter(BasePlatformAdapter):
        def __init__(self, adapter_config: Any):
            super().__init__(adapter_config, Platform.FEISHU)
            self.captured_operations: list[dict[str, Any]] = []
            self._mark_connected()

        async def connect(self) -> bool:
            self._mark_connected()
            return True

        async def disconnect(self) -> None:
            self._mark_disconnected()

        async def send(
            self,
            chat_id: str,
            content: str,
            reply_to: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "text",
                    "chat_id": chat_id,
                    "content": str(content or ""),
                    "reply_to": reply_to,
                    "metadata": dict(metadata or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def edit_message(self, chat_id: str, message_id: str, content: str) -> Any:
            self.captured_operations.append(
                {
                    "kind": "edit",
                    "chat_id": chat_id,
                    "content": str(content or ""),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_image(
            self,
            chat_id: str,
            image_url: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "image_url",
                    "chat_id": chat_id,
                    "image_url": str(image_url or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(metadata or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_voice(
            self,
            chat_id: str,
            audio_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "audio_file",
                    "chat_id": chat_id,
                    "file_path": str(audio_path or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_video(
            self,
            chat_id: str,
            video_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "video_file",
                    "chat_id": chat_id,
                    "file_path": str(video_path or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_document(
            self,
            chat_id: str,
            file_path: str,
            caption: Optional[str] = None,
            file_name: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "document_file",
                    "chat_id": chat_id,
                    "file_path": str(file_path or ""),
                    "file_name": str(file_name or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def send_image_file(
            self,
            chat_id: str,
            image_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            message_id = f"capture:{uuid.uuid4().hex[:12]}"
            self.captured_operations.append(
                {
                    "kind": "image_file",
                    "chat_id": chat_id,
                    "file_path": str(image_path or ""),
                    "caption": str(caption or ""),
                    "reply_to": reply_to,
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "message_id": message_id,
                }
            )
            return SendResult(success=True, message_id=message_id)

        async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
            normalized_chat_id = str(chat_id or "").strip()
            return {
                "chat_id": normalized_chat_id,
                "name": normalized_chat_id or "Feishu Chat",
                "type": "dm",
            }

    return _FeishuInternalCaptureAdapter(config)
