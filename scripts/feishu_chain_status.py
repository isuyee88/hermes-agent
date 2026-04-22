from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACT_FILENAMES = {
    "sessions_json": "tmp-sessions-latest.json",
    "agent_log": "tmp-modal-agent-latest.log",
    "errors_log": "tmp-modal-errors-latest.log",
}

ASSISTANT_PROVIDER_CONFIG_ERROR = "No inference provider configured"

SEND_COMPLETE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).+?"
    r"\[Feishu\] Background task send complete session=(?P<session>\S+) "
    r"message_id=(?P<message_id>\S+) success=(?P<success>\w+) "
    r"send_message_id=(?P<send_message_id>\S+) error=(?P<error>.*?)(?:\s+send_elapsed_ms=(?P<send_elapsed_ms>\d+))?$"
)
RESPONSE_READY_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).+?"
    r"response ready: platform=feishu chat=(?P<chat_id>\S+) time=(?P<time_seconds>[0-9.]+)s "
    r"api_calls=(?P<api_calls>\d+) response=(?P<response_chars>\d+) chars"
)


@dataclass
class SessionSummary:
    session_key: str
    session_id: str
    updated_at: str
    display_name: str
    platform: str
    chat_type: str
    chat_id: str
    user_id: str
    provider: str | None
    model: str | None


@dataclass
class TranscriptSummary:
    transcript_path: str | None
    assistant_message_count: int
    last_assistant_preview: str
    provider_auth_failed: bool


@dataclass
class ResponseReadySummary:
    timestamp: str
    chat_id: str
    time_seconds: float
    api_calls: int
    response_chars: int


@dataclass
class DeliverySummary:
    timestamp: str
    session_key: str
    message_id: str
    success: bool
    send_message_id: str | None
    error: str
    send_elapsed_ms: int | None


@dataclass
class ChainStatus:
    status: str
    blocker: str
    session: SessionSummary | None
    transcript: TranscriptSummary | None
    response_ready: ResponseReadySummary | None
    delivery: DeliverySummary | None


def _trim(value: Any) -> str:
    return str(value or "").strip()


def _parse_timestamp(value: str) -> datetime:
    normalized = _trim(value)
    if not normalized:
        return datetime.min
    if normalized.endswith("Z"):
        normalized = normalized.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        try:
            return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            return datetime.min


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _resolve_artifact_path(explicit: str | None, *, artifacts_dir: Path, key: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    candidate = artifacts_dir / DEFAULT_ARTIFACT_FILENAMES[key]
    return candidate if candidate.exists() else None


def _run_modal_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _modal_get_file(volume_name: str, remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run_modal_command(
        [
            "modal",
            "volume",
            "get",
            "--force",
            volume_name,
            remote_path,
            str(local_path),
        ]
    )
    return result.returncode == 0 and local_path.exists()


def _pull_modal_artifacts(
    *,
    volume_name: str,
    download_dir: Path,
    chat_id: str,
    session_key: str,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    sessions_json = download_dir / DEFAULT_ARTIFACT_FILENAMES["sessions_json"]
    agent_log = download_dir / DEFAULT_ARTIFACT_FILENAMES["agent_log"]
    errors_log = download_dir / DEFAULT_ARTIFACT_FILENAMES["errors_log"]

    sessions_ok = _modal_get_file(volume_name, "/hermes-home/sessions/sessions.json", sessions_json)
    _modal_get_file(volume_name, "/hermes-home/logs/agent.log", agent_log)
    _modal_get_file(volume_name, "/hermes-home/logs/errors.log", errors_log)

    transcript_path: Path | None = None
    if sessions_ok:
        session = load_latest_feishu_session(sessions_json, chat_id=chat_id, session_key=session_key)
        if session and session.session_id:
            candidate = download_dir / f"{session.session_id}.jsonl"
            if _modal_get_file(
                volume_name,
                f"/hermes-home/sessions/{session.session_id}.jsonl",
                candidate,
            ):
                transcript_path = candidate

    return (
        sessions_json if sessions_json.exists() else None,
        agent_log if agent_log.exists() else None,
        errors_log if errors_log.exists() else None,
        transcript_path,
    )


def load_latest_feishu_session(
    sessions_path: Path,
    *,
    chat_id: str = "",
    session_key: str = "",
) -> SessionSummary | None:
    payload = _read_json(sessions_path)
    if not isinstance(payload, dict):
        return None

    candidates: list[tuple[datetime, SessionSummary]] = []
    for raw_session_key, raw_value in payload.items():
        if not isinstance(raw_value, dict):
            continue
        platform = _trim(raw_value.get("platform"))
        if platform != "feishu":
            continue
        origin = raw_value.get("origin") if isinstance(raw_value.get("origin"), dict) else {}
        session = SessionSummary(
            session_key=_trim(raw_value.get("session_key") or raw_session_key),
            session_id=_trim(raw_value.get("session_id")),
            updated_at=_trim(raw_value.get("updated_at")),
            display_name=_trim(raw_value.get("display_name")),
            platform=platform,
            chat_type=_trim(raw_value.get("chat_type")),
            chat_id=_trim(origin.get("chat_id")),
            user_id=_trim(origin.get("user_id")),
            provider=_trim((raw_value.get("route_lease") or {}).get("provider")) or None,
            model=_trim((raw_value.get("route_lease") or {}).get("model")) or None,
        )
        if session_key and session.session_key != session_key:
            continue
        if chat_id and session.chat_id != chat_id:
            continue
        candidates.append((_parse_timestamp(session.updated_at), session))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_transcript_summary(transcript_path: Path | None) -> TranscriptSummary | None:
    if transcript_path is None or not transcript_path.exists():
        return None

    assistant_messages: list[str] = []
    provider_auth_failed = False
    for record in _iter_jsonl(transcript_path):
        role = _trim(record.get("role"))
        content = _trim(record.get("content"))
        if role == "assistant":
            assistant_messages.append(content)
            if ASSISTANT_PROVIDER_CONFIG_ERROR in content:
                provider_auth_failed = True

    last_assistant_preview = assistant_messages[-1][:240] if assistant_messages else ""
    return TranscriptSummary(
        transcript_path=str(transcript_path),
        assistant_message_count=len(assistant_messages),
        last_assistant_preview=last_assistant_preview,
        provider_auth_failed=provider_auth_failed,
    )


def _scan_latest_response_ready(log_path: Path | None, *, chat_id: str = "") -> ResponseReadySummary | None:
    if log_path is None or not log_path.exists():
        return None

    latest: tuple[datetime, ResponseReadySummary] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RESPONSE_READY_RE.search(line)
        if not match:
            continue
        if chat_id and match.group("chat_id") != chat_id:
            continue
        summary = ResponseReadySummary(
            timestamp=match.group("timestamp"),
            chat_id=match.group("chat_id"),
            time_seconds=float(match.group("time_seconds")),
            api_calls=int(match.group("api_calls")),
            response_chars=int(match.group("response_chars")),
        )
        stamp = _parse_timestamp(summary.timestamp)
        if latest is None or stamp >= latest[0]:
            latest = (stamp, summary)
    return latest[1] if latest else None


def _scan_latest_delivery(log_path: Path | None, *, session_key: str = "") -> DeliverySummary | None:
    if log_path is None or not log_path.exists():
        return None

    latest: tuple[datetime, DeliverySummary] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SEND_COMPLETE_RE.search(line)
        if not match:
            continue
        if session_key and match.group("session") != session_key:
            continue
        error = _trim(match.group("error"))
        send_message_id = _trim(match.group("send_message_id"))
        summary = DeliverySummary(
            timestamp=match.group("timestamp"),
            session_key=match.group("session"),
            message_id=match.group("message_id"),
            success=match.group("success").lower() == "true",
            send_message_id=send_message_id if send_message_id not in {"", "None", "null"} else None,
            error="" if error in {"", "None", "null"} else error,
            send_elapsed_ms=int(match.group("send_elapsed_ms")) if match.group("send_elapsed_ms") else None,
        )
        stamp = _parse_timestamp(summary.timestamp)
        if latest is None or stamp >= latest[0]:
            latest = (stamp, summary)
    return latest[1] if latest else None


def _classify_blocker(
    *,
    transcript: TranscriptSummary | None,
    delivery: DeliverySummary | None,
    response_ready: ResponseReadySummary | None,
) -> str:
    if delivery and delivery.success:
        return "none"
    if transcript and transcript.provider_auth_failed:
        return "provider_not_configured"
    if delivery and "99992354" in delivery.error:
        return "invalid_reply_to"
    if delivery and "bot/user can not be out of the chat" in delivery.error.lower():
        return "bot_not_in_target_chat"
    if response_ready and not delivery:
        return "send_not_observed"
    if transcript and transcript.assistant_message_count == 0:
        return "assistant_not_observed"
    return "unknown"


def analyze_chain_status(
    *,
    sessions_json: Path,
    agent_log: Path | None,
    errors_log: Path | None,
    transcript_path: Path | None,
    chat_id: str = "",
    session_key: str = "",
) -> ChainStatus:
    session = load_latest_feishu_session(sessions_json, chat_id=chat_id, session_key=session_key)
    effective_chat_id = chat_id or (session.chat_id if session else "")
    effective_session_key = session_key or (session.session_key if session else "")
    transcript = load_transcript_summary(transcript_path)
    response_ready = _scan_latest_response_ready(agent_log, chat_id=effective_chat_id) or _scan_latest_response_ready(
        errors_log, chat_id=effective_chat_id
    )
    delivery = _scan_latest_delivery(agent_log, session_key=effective_session_key) or _scan_latest_delivery(
        errors_log, session_key=effective_session_key
    )
    blocker = _classify_blocker(
        transcript=transcript,
        delivery=delivery,
        response_ready=response_ready,
    )
    status = "ok" if delivery and delivery.success else "blocked"
    return ChainStatus(
        status=status,
        blocker=blocker,
        session=session,
        transcript=transcript,
        response_ready=response_ready,
        delivery=delivery,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the latest Feishu -> Modal -> Hermes -> Feishu delivery state from saved artifacts."
    )
    parser.add_argument("--artifacts-dir", default=str(REPO_ROOT.parent))
    parser.add_argument("--sessions-json", default="")
    parser.add_argument("--agent-log", default="")
    parser.add_argument("--errors-log", default="")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--chat-id", default="")
    parser.add_argument("--session-key", default="")
    parser.add_argument("--pull-from-modal", action="store_true")
    parser.add_argument("--modal-volume", default="hermes-agent-data")
    parser.add_argument("--download-dir", default="")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    transcript_path = Path(args.transcript).resolve() if args.transcript else None
    if args.pull_from_modal:
        download_dir = Path(args.download_dir).resolve() if args.download_dir else artifacts_dir
        sessions_json, agent_log, errors_log, modal_transcript = _pull_modal_artifacts(
            volume_name=_trim(args.modal_volume) or "hermes-agent-data",
            download_dir=download_dir,
            chat_id=_trim(args.chat_id),
            session_key=_trim(args.session_key),
        )
        if transcript_path is None:
            transcript_path = modal_transcript
    else:
        sessions_json = _resolve_artifact_path(args.sessions_json, artifacts_dir=artifacts_dir, key="sessions_json")
        agent_log = _resolve_artifact_path(args.agent_log, artifacts_dir=artifacts_dir, key="agent_log")
        errors_log = _resolve_artifact_path(args.errors_log, artifacts_dir=artifacts_dir, key="errors_log")

    if sessions_json is None:
        raise SystemExit("Missing sessions artifact. Pass --sessions-json or place tmp-sessions-latest.json in --artifacts-dir.")

    status = analyze_chain_status(
        sessions_json=sessions_json,
        agent_log=agent_log,
        errors_log=errors_log,
        transcript_path=transcript_path,
        chat_id=_trim(args.chat_id),
        session_key=_trim(args.session_key),
    )
    payload = asdict(status)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if status.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
