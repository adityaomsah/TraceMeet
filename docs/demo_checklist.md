# Demo and release checklist

## Record the demonstration

Use a recording you may redistribute. Close terminals containing credentials.
Show the following in a concise screen recording:

1. Introduce TraceMeet and identify the three model roles.
2. Upload a new English recording; enter optional glossary/protected names.
3. Click Process recording and show stage status. If speeding up the wait, label
   the edit and state the actual elapsed time; do not imply real-time inference.
4. Show raw and refined wording and one held correction if present in that run.
5. Show summary, minutes, decisions and tasks. Explain proposals versus agreement,
   tentative tasks, and Unspecified owners/deadlines where present.
6. Click a citation and play the corresponding source segment.
7. Download/open the ZIP and show consistent Markdown, JSON and task CSV.
8. Reopen through Saved runs. Explain that viewing makes no inference calls;
   resuming invalidated work can consume quota.
9. State the key limitation: exact citation matching does not prove entailment.

Do not fabricate an edit, task or failure if the chosen recording does not contain
one. A different genuine run may be shown if the switch is explicit.

## Sample package

For each published sample, include the exact recording and its real generated
raw/refined transcripts, meeting record, readable minutes, task CSV and relevant
audit reports. Record source, clip interval, redistribution permission/license,
model/configuration and run ID. Do not pair a new clip with outputs from the
original full-length recording.

## Before submission

- Confirm project license, sample permission and repository visibility.
- Replace pending demo/sample text in README with actual relative links.
- Run a fresh clone and Python 3.11 venv install using the committed lock file.
- Run pip check, the full test suite, offline and live preflight.
- Process a new short recording and verify downloads/playback.
- Verify bad media, missing keys and resumable failure behavior without damaging
  the published sample. Never deliberately exhaust API quota.
- Check the actual submission requirements and final deadline.
- Review git status and staged files; exclude .env, private runs and venv contents.
- Record the final commit hash and submit early enough to verify receipt.

Evaluation and the full LaTeX report must be completed or clearly identified as
pending; do not present implementation tests as measured model accuracy.
