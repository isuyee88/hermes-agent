from __future__ import annotations

import asyncio
from typing import Any, Callable


def get_named_queue(
    *,
    modal_module: Any,
    current_queue: Any,
    queue_name: str,
    missing_message: str,
) -> Any:
    if modal_module is None:
        raise RuntimeError(missing_message)
    if current_queue is None:
        return modal_module.Queue.from_name(queue_name, create_if_missing=True)
    return current_queue


async def prewarm_queue_len(
    *,
    queue: Any,
    is_prewarmed: bool,
    set_prewarmed: Callable[[bool], None],
    logger: Any,
    warning_prefix: str,
) -> None:
    if is_prewarmed:
        return
    try:
        if hasattr(queue, "len") and hasattr(queue.len, "aio"):
            await queue.len.aio()  # type: ignore[union-attr]
        else:
            await asyncio.to_thread(queue.len)
        set_prewarmed(True)
    except Exception as exc:
        logger.warning("%s: %s", warning_prefix, exc)


def sync_modal_volume(*, volume: Any, reload: bool, commit: bool, logger: Any, warning_prefix: str) -> None:
    if volume is None:
        return
    try:
        if reload:
            volume.reload()
        if commit:
            volume.commit()
    except Exception as exc:
        logger.warning("%s (reload=%s commit=%s): %s", warning_prefix, reload, commit, exc)


async def sync_modal_volume_async(*, volume: Any, reload: bool, commit: bool, logger: Any, warning_prefix: str) -> None:
    if volume is None:
        return
    try:
        if reload:
            reload_handle = getattr(volume, "reload", None)
            if reload_handle is not None and hasattr(reload_handle, "aio"):
                await reload_handle.aio()  # type: ignore[union-attr]
            else:
                await asyncio.to_thread(volume.reload)
        if commit:
            commit_handle = getattr(volume, "commit", None)
            if commit_handle is not None and hasattr(commit_handle, "aio"):
                await commit_handle.aio()  # type: ignore[union-attr]
            else:
                await asyncio.to_thread(volume.commit)
    except Exception as exc:
        logger.warning("%s (reload=%s commit=%s): %s", warning_prefix, reload, commit, exc)


def safe_queue_depth(*, get_queue: Callable[[], Any], logger: Any, warning_prefix: str) -> int | None:
    try:
        return int(get_queue().len())
    except Exception as exc:
        logger.warning("%s: %s", warning_prefix, exc)
        return None


async def safe_queue_depth_async(*, queue: Any, logger: Any, warning_prefix: str) -> int | None:
    try:
        if hasattr(queue, "len") and hasattr(queue.len, "aio"):
            return int(await queue.len.aio())  # type: ignore[union-attr]
        return int(await asyncio.to_thread(queue.len))
    except Exception as exc:
        logger.warning("%s: %s", warning_prefix, exc)
        return None
