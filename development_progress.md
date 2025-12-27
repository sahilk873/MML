# Development Progress

## Current snapshot

- **Stage 1 – KG ingestion**: `run_project.py` now discovers, cleans, and caches all `part-*.snappy.parquet` pieces while logging basic edge/entity counts.
- **Stage 2 – Label ingestion & normalization**: positive/negative CSVs are read, duplicated pairs are canonicalized, conflicts favor positives, and `labels_clean.parquet` is persisted.
- **Stage 3 – Coverage filtering**: labels are aligned with KG entities, missing rows are recorded in `dropped_missing_entities.csv`, and the filtered set is saved.
- **Stage 4 – ID mappings**: entity/relation maps are built from the KG and saved as `entity2id.json`/`relation2id.json`.
- **Stage 5 – KG embeddings**: PyKEEN training is invoked via `train_pykeen`, and the new CLI arguments (`--embed_epochs`, `--embed_batch_size`, `--embed_lr`) make it easy to control the job for future resumptions.
- **Stage 6 – Splits & classifier**: random and pair holdout splits feed the feature builder, MLP training, ablations, and prediction writing, with per-stage progress recorded in `artifacts/progress.json`.
- **Stage 7 – Reporting**: metrics/report/prediction outputs (plus the optional ablations) are generated and logged; `progress.json` now captures stage completion timestamps.
- **Stage 8 – Modular kg_project layout & smoke/unit tests**: created `kg_project/{io,preprocess,kg_embed,features,splits,model,eval}` modules, rewired `run_project.py` to orchestrate through them, and added smoke/unit tests under `kg_project/tests`.

## Next steps

1. Gradually refactor the code into the modular `kg_project/` package layout described in `agent.md` so each stage (I/O, preprocess, embeddings, features, splits, model, eval) lives in its own module.
2. Add the requested smoke/unit tests (`kg_project/tests/…`) and connect them to the `--run_tests` flag so the pipeline can validate itself before every run.
3. Execute the pipeline with tuned epoch/batch settings, examine the generated artifacts, and capture any insights/failures in `report.txt` before the next pause.
4. Troubleshoot the `python -m pytest` crash: it aborts when imported modules pull in `torch`, so we need a workaround or environment note before relying on the test suite.
