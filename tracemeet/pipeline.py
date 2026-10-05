import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from tracemeet.audio.validate import validate_media
from tracemeet.guards.diff import edit_metrics, transcript_edits
from tracemeet.guards.evidence import EvidenceResult, verify_evidence
from tracemeet.guards.sensitive import GuardResult, guard_refinement
from tracemeet.schemas import MeetingRecord, Transcript
from tracemeet.stages.minutes import generate_minutes
from tracemeet.stages.refine import refine_transcript

ROOT = Path(__file__).resolve().parent.parent

DEPENDENCIES = [
    "prompts/refine_v1.txt",
    "prompts/minutes_v1.txt",
    "tracemeet/schemas.py",
    "tracemeet/pipeline.py",
    "tracemeet/stt/local_whisper.py",
    "tracemeet/audio/validate.py",
    "tracemeet/stages/refine.py",
    "tracemeet/stages/minutes.py",
    "tracemeet/guards/diff.py",
    "tracemeet/guards/sensitive.py",
    "tracemeet/guards/evidence.py",
    "tracemeet/llm/base.py",
    "tracemeet/llm/gemini.py",
    "tracemeet/llm/router.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_json(path: Path, data: dict) -> None:
    save_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def settings_signature(cfg: dict, terms: str, names: list[str]) -> str:
    payload = {
        "config": cfg,
        "terms": terms,
        "protected_names": names,
        "dependencies": {
            relative: file_hash(ROOT / relative)
            for relative in DEPENDENCIES
        },
    }
    return text_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))


@dataclass
class Run:
    directory: Path
    media_path: Path
    original_filename: str
    cfg: dict
    terms: str
    names: list[str]
    signature: str
    media_sha256: str

    raw: Transcript | None = None
    candidate: Transcript | None = None
    guarded: GuardResult | None = None
    candidate_record: MeetingRecord | None = None
    checked: EvidenceResult | None = None

    status: str = "ready"
    stage: str = "Preparation"
    error: str | None = None
    seconds: dict[str, float] = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)


def save_run_state(run: Run) -> None:
    save_json(
        run.directory / "run_state.json",
        {
            "status": run.status,
            "stage": run.stage,
            "error": run.error,
            "settings_signature": run.signature,
            "media_sha256": run.media_sha256,
            "stage_seconds": run.seconds,
            "completed": {
                "transcription": run.raw is not None,
                "refinement": run.candidate is not None,
                "guards": run.guarded is not None,
                "minutes": run.candidate_record is not None,
                "evidence": run.checked is not None,
            },
            "attempts": run.attempts,
        },
    )


def create_run(
    directory: Path,
    media_path: Path,
    original_filename: str,
    cfg: dict,
    terms: str,
    names: list[str],
) -> Run:
    # Copy settings; do not store API keys in cfg.
    cfg_copy = json.loads(json.dumps(cfg))
    run = Run(
        directory=directory,
        media_path=media_path,
        original_filename=original_filename,
        cfg=cfg_copy,
        terms=terms,
        names=list(names),
        signature=settings_signature(cfg_copy, terms, names),
        media_sha256=file_hash(media_path),
    )

    save_json(
        directory / "meta.json",
        {
            "original_filename": original_filename,
            "terms": terms,
            "protected_names": names,
            "config": cfg_copy,
            "created": utc_now(),
            "media_sha256": run.media_sha256,
        },
    )
    save_run_state(run)
    return run


def continue_run(
    run: Run,
    *,
    provider,
    engine_factory: Callable,
    on_status: Callable[[str], None],
    on_progress: Callable[[float], None],
) -> None:
    if settings_signature(run.cfg, run.terms, run.names) != run.signature:
        raise ValueError(
            "Code, prompts, or settings changed. Start a new run."
        )

    if file_hash(run.media_path) != run.media_sha256:
        raise ValueError("The saved recording changed. Start a new run.")

    attempt = {
        "id": uuid4().hex,
        "started": utc_now(),
        "status": "running",
    }
    run.attempts.append(attempt)
    run.status = "running"
    run.error = None

    def begin(stage: str) -> float:
        run.stage = stage
        save_run_state(run)
        on_status(stage)
        return perf_counter()

    def finished(stage: str, started: float) -> None:
        run.seconds[stage] = perf_counter() - started
        save_run_state(run)

    try:
        if run.raw is None:
            started = begin("Transcription")
            validate_media(run.media_path)
            engine = engine_factory()
            raw = engine.transcribe(
                run.media_path,
                terms=run.terms or None,
                on_progress=on_progress,
            )
            save_text(
                run.directory / "raw_transcript.json",
                raw.model_dump_json(indent=2),
            )
            run.raw = raw
            finished("Transcription", started)

        if run.candidate is None:
            started = begin("Refinement")
            candidate = refine_transcript(
                run.raw,
                provider,
                model=run.cfg["llm"]["refine_model"],
                temperature=run.cfg["llm"]["temperature"],
                glossary=run.terms,
                checkpoint_dir=run.directory / "refinement_chunks",
                on_status=on_status,
            )
            edits = transcript_edits(run.raw, candidate)
            save_text(
                run.directory / "candidate_refined_transcript.json",
                candidate.model_dump_json(indent=2),
            )
            save_json(
                run.directory / "candidate_edits.json",
                {"status": "candidate_not_guarded", "edits": edits},
            )
            save_json(
                run.directory / "refinement_meta.json",
                {
                    "status": "candidate_not_guarded",
                    "provider": provider.name,
                    "model": run.cfg["llm"]["refine_model"],
                    "temperature": run.cfg["llm"]["temperature"],
                    "glossary": run.terms,
                    "source_sha256": file_hash(
                        run.directory / "raw_transcript.json"
                    ),
                    "prompt_sha256": file_hash(ROOT / "prompts/refine_v1.txt"),
                    "metrics": edit_metrics(edits),
                    "created": utc_now(),
                },
            )
            run.candidate = candidate
            finished("Refinement", started)

        if run.guarded is None:
            started = begin("Sensitive-change checks")
            guarded = guard_refinement(
                run.raw, run.candidate, protected_names=run.names
            )
            save_text(
                run.directory / "refined_transcript.json",
                guarded.transcript.model_dump_json(indent=2),
            )
            save_json(
                run.directory / "guard_report.json",
                {
                    "protected_names": run.names,
                    "raw_sha256": file_hash(
                        run.directory / "raw_transcript.json"
                    ),
                    "candidate_sha256": file_hash(
                        run.directory / "candidate_refined_transcript.json"
                    ),
                    "corrections": [
                        item.model_dump(mode="json")
                        for item in guarded.corrections
                    ],
                    "edits": guarded.edits,
                },
            )
            run.guarded = guarded
            finished("Sensitive-change checks", started)

        if run.candidate_record is None:
            started = begin("Meeting documentation")
            candidate_record = generate_minutes(
                run.guarded.transcript,
                provider,
                model=run.cfg["llm"]["minutes_model"],
                temperature=run.cfg["llm"]["temperature"],
                on_status=on_status,
            )
            record_json = candidate_record.model_dump_json(indent=2)
            save_text(
                run.directory / "candidate_meeting_record.json",
                record_json,
            )
            save_json(
                run.directory / "minutes_meta.json",
                {
                    "status": "candidate_evidence_unverified",
                    "provider": provider.name,
                    "model": run.cfg["llm"]["minutes_model"],
                    "temperature": run.cfg["llm"]["temperature"],
                    "source_sha256": file_hash(
                        run.directory / "refined_transcript.json"
                    ),
                    "record_sha256": text_hash(record_json),
                    "prompt_sha256": file_hash(ROOT / "prompts/minutes_v1.txt"),
                    "created": utc_now(),
                },
            )
            run.candidate_record = candidate_record
            finished("Meeting documentation", started)

        if run.checked is None:
            started = begin("Citation checks")
            checked = verify_evidence(
                run.candidate_record, run.guarded.transcript
            )
            record_json = checked.record.model_dump_json(indent=2)
            checked.report.update(
                source_sha256=file_hash(
                    run.directory / "refined_transcript.json"
                ),
                candidate_sha256=file_hash(
                    run.directory / "candidate_meeting_record.json"
                ),
                checked_record_sha256=text_hash(record_json),
            )
            save_text(
                run.directory / "citation_checked_meeting_record.json",
                record_json,
            )
            save_json(run.directory / "evidence_report.json", checked.report)
            run.checked = checked
            finished("Citation checks", started)

        run.status = "complete"
        run.stage = "Complete"
        attempt["status"] = "complete"

    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)
        attempt.update(status="failed", failed_stage=run.stage)
        raise

    finally:
        attempt["finished"] = utc_now()
        save_run_state(run)