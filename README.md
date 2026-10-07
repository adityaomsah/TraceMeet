# TraceMeet

**Meeting records you can verify.**

TraceMeet turns a recorded English meeting into a raw transcript, a domain-aware refined transcript, and a structured meeting record: summary, minutes, decisions, action items and open questions. Every item cites the transcript segments it came from, and missing owners or deadlines are shown as **Unspecified**, never guessed.

> **Status: draft.** Sections marked `TODO` depend on the final pipeline and must be completed before submission.

---

## How it works

1. **Validate** the upload (opens as audio or video, has an audio track, is not empty or too short).
2. **Transcribe** locally with faster-whisper. The raw transcript is saved and never modified.
3. **Refine** domain terms with a language model (LLM 1). The refiner proposes whole-segment rewrites, and the program computes the word-level differences itself.
4. **Guard** the edits. Changes that touch numbers, names, negation, commitments, dates, units or currencies are held for review, and the raw wording is kept.
5. **Document** the meeting with a separate language model (LLM 2): summary, minutes, decisions, tasks, open questions. Long meetings are processed in overlapping windows and then merged.
6. **Verify citations.** Each cited quote must appear in the cited segment. Unsupported items are dropped, and unsupported owners or deadlines are cleared.
7. **Export** Markdown, JSON, task CSV and a ZIP, all generated from the same validated record.

## Models

| Stage | Model | Where it runs | Role |
|---|---|---|---|
| Speech-to-text | faster-whisper `small.en` (CPU, int8) | Locally | Timestamped raw transcript |
| LLM 1: refinement | `TODO: provider and model ID` | Cloud API | Corrects domain-specific recognition errors, preserves meaning |
| LLM 2: minutes | `TODO: provider and model ID` | Cloud API | Minutes, decisions, action items, open questions |

The two language models are separate models used in separate stages.

## Requirements

- Windows 10 or 11 (the setup below uses PowerShell)
- Python 3.11
- Internet access: the first run downloads the speech model, and the language-model stages call cloud APIs
- `TODO: which API key(s) are needed and where to create them (free tiers are enough for the sample)`
- No GPU is needed. Transcription runs on the CPU.

## Setup

```powershell
git clone https://github.com/adityaomsah/TraceMeet.git
cd TraceMeet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and add your key(s): `TODO: list the variable names`.

Check the setup (this makes a few small API requests):

```powershell
python -m scripts.preflight
```

Start the app:

```powershell
streamlit run app.py
```

or run `scripts\run.ps1`. If PowerShell blocks the script, use `powershell -ExecutionPolicy Bypass -File scripts\run.ps1`.

## Using the app

1. Upload an English meeting recording (audio or video). Files up to 2 GB are accepted.
2. Optionally list participant names and key terms. They give the speech model context, but they are not proof that a name was spoken.
3. Click **Transcribe recording** and, `TODO: describe the processing button once the pipeline is wired in`.
4. Inspect the raw and refined transcripts, the correction log, and the meeting record. Click a citation to play the recording from that point.
5. Download the results.

The **Saved runs** page reopens any earlier run from the `runs/` folder without calling any API.

## Outputs

| Output | Contents |
|---|---|
| Raw transcript | Speech-to-text result with timestamps, unchanged |
| Refined transcript | Transcript after guarded domain-term correction |
| Correction log | Every changed segment, shown word by word |
| Meeting record | Summary, minutes, decisions, action items, open questions, each with citations |
| Markdown, JSON, CSV, ZIP | The same record in human-readable and machine-readable form |

Decisions carry a status: `decided`, `proposed`, `rejected` or `unresolved`. Tasks carry `confirmed` or `tentative`. An owner or deadline appears only when the recording states it and a cited quote supports it.

## Troubleshooting

| Problem | What to do |
|---|---|
| First run is slow | The speech model is downloading once. Later runs reuse it. |
| Hugging Face symlink warning | Harmless on Windows. Caching still works. |
| `GEMINI_API_KEY` or other key is not set | `TODO: match the final variable names.` Copy `.env.example` to `.env` and add the key. |
| Rate limit or quota message | Free-tier limits are set by the provider and can change. Check the limits shown in the provider's console, wait, and run again. Finished stages are kept in the run folder. |
| "Temporarily unavailable" or timeout | The provider is overloaded or the request timed out. Try again later. |
| "No audio track" | The file has no audio stream. Upload a recording that contains sound. |
| "No speech was transcribed" | The recording is silent or the speech is inaudible. |
| Upload larger than the limit | The limit is set in `.streamlit/config.toml` (`maxUploadSize`, in MB). |
| Port 8501 is in use | Run `streamlit run app.py --server.port 8502`. |

## Privacy

Audio is transcribed locally. **The transcript text is sent to cloud language-model providers** (`TODO: name them`) for refinement and documentation. Free-tier terms may allow providers to use submitted content to improve their products. Do not upload recordings that are confidential or that you are not allowed to share.

## Limitations

- English only.
- Names that you do not supply can be misspelled by the speech model. TraceMeet does not guess them, and edits that touch names are held for review.
- A citation check shows that a quoted passage exists in the cited segment. It does not prove that the claim follows from the passage.
- Condensing a long meeting into a summary can soften or merge nuance. `TODO: describe the measured behaviour after the final evaluation`.
- Cloud model output can vary between runs.
- Free-tier quotas change without notice.
- Long recordings take a while to transcribe on a CPU. Progress is shown.
- Streamlit holds an upload in memory, so very large files may be slow.

## Evaluation

`TODO: results table (word error rate, decision and task precision and recall, unsupported owners and deadlines, raw versus refined), the recordings used, and what was held out.`

## Repository layout

```
app.py               Streamlit entry point
pages/               Additional Streamlit pages (saved runs)
tracemeet/           Application code: audio, stt, llm, stages, guards, export, ui
prompts/             Versioned model instructions
config/              Model and settings configuration
scripts/             Preflight check, development tools, launcher
tests/               Automated tests
eval/                Evaluation data and scripts
samples/             Shareable sample recording and its generated outputs
docs/                Technical description
```

## Tests

```powershell
python -m pytest tests -q
```

The tests cover validation, schemas, guards, citation checks and exports. They show that the tested cases behave as intended. They are not a measure of transcription or summarisation accuracy.

## License

`TODO: choose and add a LICENSE file.`
