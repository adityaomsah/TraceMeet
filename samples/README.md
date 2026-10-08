# TraceMeet — sample recordings and generated outputs

This package pairs two five-minute development recordings with their actual downloaded TraceMeet exports. The author supplied the recording-to-export mapping. The exports contain no recording hash, so this mapping has not been independently established from run metadata. Audio hashes are recorded in verification.json for future identification.

## Start here

1. Open `samples/sample_01/meeting_record.md` for readable output from sample 1, or the equivalent file under `sample_02`.
2. Open `tasks.csv` in a spreadsheet for extracted tasks.
3. Compare `raw_transcript.json` and `refined_transcript.json` using stable segment IDs.
4. Read `evidence_report.json` for citation results and limitations; listen to the matching WAV around the cited segment timestamps.
5. To generate a new result, upload either WAV through the app and click **Process recording**. Provider access and quota are required; regenerated outputs may differ.

| Sample | Audio duration | Segments | Minutes topics | Decisions | Tasks | Open questions | Exact citation matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 s | 91 | 15 | 1 | 1 | 0 | 53 |
| 2 | 300 s | 68 | 5 | 0 | 1 | 1 | 19 |

Counts describe generated items and citation occurrences, not accuracy, unique supporting passages or independently confirmed decisions. Read the decision/task status fields before treating any item as agreed.

## What was checked

Both original export manifests match their file hashes. Raw/refined segment IDs and timestamps agree. Record counts match their manifests; task CSV fields and evidence match JSON with the documented spreadsheet-safe formatting. All 72 citation occurrences across the two exports pass exact source-substring and word-boundary checks. These checks do not establish that the claims follow from the speech.

The exports were copied without changing their content. The original manifest.json covers the original export files only; PACKAGE_SHA256.json covers package files including audio and this guide. Neither manifest is a digital signature.

The author reported **246 passing tests in 6.33 seconds at commit 7014be9**. This is not a new test run performed while packaging. The exported STT identifier is faster-whisper/small.en. Provider/model metadata for language stages is absent from these exports; the source configuration uses Gemini refinement and Groq GPT-OSS minutes, but per-run provider provenance would require the original run metadata.

## Source and sharing

These are excerpts from the same third-party recording, not separate held-out meetings. The development source link supplied by the author is https://www.youtube.com/watch?v=rOqgRiNMVqg. The exact clip offsets and redistribution permission have not been documented here. Confirm permission or an appropriate source license before publishing the recordings or their transcripts. The project's MIT code license does not automatically license the recording content.

The public technical report uses aggregate counts and synthetic examples rather than reproducing meeting names or transcript passages. Guard correction reports and the model's proposed transcript are not included in these exports; obtain them from the original run when needed.
