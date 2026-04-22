from __future__ import annotations

from typing import Any, Callable


def process_cron_queue_item(
    payload: Any,
    *,
    prepare_runtime_environment: Callable[[], None],
    should_reload_modal_volume_for_claims: Callable[[str], bool],
    sync_modal_volume: Callable[..., None],
    should_process_queued_cron_job: Callable[[dict[str, Any] | None, Any], tuple[bool, str]],
    release_cron_job_claim: Callable[..., None],
    logger: Any,
) -> dict[str, Any]:
    prepare_runtime_environment()
    if should_reload_modal_volume_for_claims("cron"):
        sync_modal_volume(reload=True)

    from cron.jobs import get_job, mark_job_run, save_job_output
    from cron.scheduler import SILENT_MARKER, _deliver_result, run_job

    if not isinstance(payload, dict):
        payload = {"job_id": str(payload or "")}

    job_id = str(payload.get("job_id") or "").strip()
    claim_token = str(payload.get("claim_token") or "").strip() or None
    if not job_id:
        return {"status": "skipped", "reason": "missing_job_id"}

    job = get_job(job_id)
    should_run, skip_reason = should_process_queued_cron_job(job, payload)
    if not should_run:
        release_cron_job_claim(job_id, claim_token=claim_token)
        return {"status": "skipped", "job_id": job_id, "reason": skip_reason}

    success, output, final_response, error = run_job(job)
    output_file = str(save_job_output(job_id, output))

    delivery_error = None
    deliver_content = (
        final_response
        if success
        else f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛闂備浇顕栭崰鏍ь焽閳ュ磭鏆﹂柟顖炲亰濡茬兘姊洪崫鍕靛剱闁绘濞€瀵鏁撻悩鑼€為梺瀹犳〃閻掞箓鎮楁繝姘拺闁硅偐鍋涙俊鑺ヤ繆椤愩垹鏆ｆ鐐插暙铻栭柛娑卞櫘濡啫鈹戦悙鏉戠仸闁荤噦绠撳畷浼村箛椤斿墽锛?Cron job '{job.get('name', job_id)}' failed:\n{error}"
    )
    should_deliver = bool(deliver_content)
    if should_deliver and success and SILENT_MARKER in deliver_content.strip().upper():
        should_deliver = False

    if should_deliver:
        try:
            delivery_error = _deliver_result(job, deliver_content)
        except Exception as exc:
            delivery_error = str(exc)
            logger.error("Cron delivery failed for job %s: %s", job_id, exc)

    mark_job_run(job_id, success, error, delivery_error=delivery_error)
    sync_modal_volume(commit=True)
    release_cron_job_claim(job_id, claim_token=claim_token)

    return {
        "status": "ok" if success else "error",
        "job_id": job_id,
        "job_name": job.get("name"),
        "output_file": output_file,
        "delivery_error": delivery_error,
        "error": error,
    }
