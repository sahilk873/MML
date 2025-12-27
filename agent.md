# agent.md — KG-informed two-component association prediction

## Objective
Build an end-to-end pipeline that uses a biomedical knowledge graph (stored as multiple Parquet parts) plus two labeled CSV files to train a model that predicts whether a triple `(chemical_1, chemical_2, target)` is a **positive association (1)** or **negative association (0)**.

**Inputs live in `./src/`** (relative to repo root).

**Engineering requirements (important):**
- Prefer **small, testable modules** over one giant script.
- Implement **unit/integration smoke tests for each stage** and run them before moving to the next stage.
- Keep I/O, pure transforms, and training code separated for easier debugging.

---

## Expected input files (in `./src/`)
- Knowledge graph Parquet parts:
  - `src/part-*.snappy.parquet` (all parts; load all matching files)
- Labeled CSVs:
  - `src/gt_two_component_positive.csv`  (label = 1)
  - `src/gt_two_component_negative.csv`  (label = 0)

---

## Output directory
Write all outputs to `./artifacts/`:

- `entity2id.json`
- `relation2id.json`
- `kg_edges.parquet` (optional cache of unified KG triples)
- `labels_clean.parquet`
- `labels_filtered.parquet`
- `dropped_missing_entities.csv`
- `node_embeddings.npy`
- `splits/random/{train,val,test}.parquet`
- `splits/pair_holdout/{train,val,test}.parquet`
- `classifier.pt`
- `metrics.json`
- `report.txt`
- `predictions_test.csv`
- `smoke_tests.log` (optional)

---

## Recommended architecture (implement this baseline first)

### Stage A — KG Embeddings (relation-aware)
Train relation-aware KG embeddings from the Parquet triples.

**Recommended models (choose one):**
- RotatE (default)
- ComplEx

**Implementation tool:** PyKEEN

**Output:** entity embeddings matrix `E` of shape `[num_entities, d]`.

### Stage B — Supervised classifier on labeled triples
For each labeled example `(c1, c2, t)`:
- Look up embeddings:
  - `e1 = E[c1_id]`
  - `e2 = E[c2_id]`
  - `et = E[t_id]`
- Build symmetric, order-invariant features:
  - `pair_sum = e1 + e2`
  - `pair_diff = abs(e1 - e2)`
  - `pair_prod = e1 * e2`
  - `c1t = e1 * et`
  - `c2t = e2 * et`
  - `pt = pair_sum * et`
- Feature vector:
  - `x = concat(pair_sum, pair_diff, pair_prod, et, c1t, c2t, pt)`

Classifier:
- MLP: `Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(1)`
- Loss: `BCEWithLogitsLoss` (use `pos_weight` or weighted sampling)
- Output: `sigmoid(logit)` as probability

---

## Code structure (modular; do NOT build a single huge script)

Create a package-style layout like:

```
.
├── run_project.py                 # thin CLI orchestrator only
├── kg_project/
│   ├── __init__.py
│   ├── io/
│   │   ├── parquet_loader.py      # load/validate KG parts
│   │   ├── csv_loader.py          # load/clean label CSVs
│   │   └── artifacts.py           # save/load artifacts paths + helpers
│   ├── preprocess/
│   │   ├── normalize_pairs.py     # enforce unordered chemical pair canonicalization
│   │   ├── coverage_filter.py     # drop label rows missing KG entities
│   │   └── mappings.py            # entity2id/relation2id builders
│   ├── kg_embed/
│   │   ├── pykeen_train.py        # train RotatE/ComplEx and export embeddings
│   │   └── subgraph.py            # optional BFS subgraph extraction
│   ├── features/
│   │   └── triple_features.py     # embed -> feature vector builder (pure functions)
│   ├── splits/
│   │   └── splitters.py           # random + pair-holdout splitting + leakage audit
│   ├── model/
│   │   ├── mlp.py                 # classifier definition
│   │   ├── dataset.py             # torch Dataset/DataLoader utilities
│   │   └── train.py               # training loop + early stopping
│   ├── eval/
│   │   ├── metrics.py             # auc/pr-auc/confusion/threshold selection
│   │   └── predict.py             # prediction writers
│   └── tests/
│       ├── test_io_smoke.py       # quick smoke tests on a small subset
│       ├── test_features.py
│       ├── test_splits.py
│       └── test_train_step.py
└── artifacts/
```

**Principles:**
- Pure transforms go in `preprocess/` and `features/` and should be easy to unit test.
- Training code should not do file globbing or string munging.
- `run_project.py` should wire stages together and be readable in one screen.

---

## Testing philosophy (must do)
For each stage below:
1) Implement it.
2) Add a **smoke test** (fast) and a **unit test** for the core logic.
3) Run tests; only then proceed.

Testing should be runnable via:
```bash
python -m pytest -q
```

If pytest is unavailable, provide a fallback `python kg_project/tests/run_smoke_tests.py` that prints PASS/FAIL.

---

## End-to-end steps (the agent must implement all)

### 1) Data ingestion (KG)
Implementation: `kg_project/io/parquet_loader.py`

1. Glob all KG parts:
   - pattern: `src/part-*.snappy.parquet`
   - sort by filename
2. Read each parquet and select columns:
   - `subject`, `predicate`, `object`
3. Concatenate into `kg_df`
4. Clean:
   - drop any rows with nulls in {subject,predicate,object}
   - cast all three columns to string
5. Log:
   - total edges (rows)
   - unique subjects, predicates, objects
   - total unique entities = unique(subject ∪ object)
6. Cache unified KG (optional but recommended):
   - write `artifacts/kg_edges.parquet`

**Tests (before moving on):**
- `test_io_smoke.py`: loads only first 1–2 parquet parts and asserts:
  - columns exist
  - non-zero row count
  - all three columns are strings after cleaning

### 2) Data ingestion (labels)
Implementation: `kg_project/io/csv_loader.py` + `kg_project/preprocess/normalize_pairs.py`

1. Read:
   - `src/gt_two_component_positive.csv` → `pos_df`, add `label=1`
   - `src/gt_two_component_negative.csv` → `neg_df`, add `label=0`
2. Keep columns:
   - `source_primary`, `source_secondary`, `target`, `label`
3. Normalize pair ordering (order-invariant):
   - set `c1 = min(source_primary, source_secondary)` (lexicographic)
   - set `c2 = max(source_primary, source_secondary)`
   - output columns should be `c1,c2,target,label`
4. Deduplicate:
   - group by `(c1,c2,target)`
   - if conflicting labels exist, keep label=1 (positive wins), log conflict count
   - drop duplicates
5. Save:
   - `artifacts/labels_clean.parquet`
6. Log:
   - total examples, positives, negatives
   - unique chemicals (from c1,c2)
   - unique targets

**Tests (before moving on):**
- `test_io_smoke.py`: asserts label file loads and has required columns.
- `test_features.py` (or separate test): asserts normalization makes `(A,B)` == `(B,A)`.

### 3) Align labels with KG entity coverage
Implementation: `kg_project/preprocess/coverage_filter.py`

1. Build entity set from KG:
   - `entities_in_kg = set(subject) ∪ set(object)`
2. Filter labeled rows:
   - drop any row where `c1` or `c2` or `target` not in `entities_in_kg`
3. Save:
   - kept rows → `artifacts/labels_filtered.parquet`
   - dropped rows with reason → `artifacts/dropped_missing_entities.csv`
4. Log:
   - number dropped, percent dropped
   - most common missing prefixes (e.g., CHEBI, DRUGBANK, MONDO, HP)

**Tests (before moving on):**
- Unit test with tiny synthetic KG and labels verifying correct keep/drop behavior.

### 4) Build integer ID mappings
Implementation: `kg_project/preprocess/mappings.py`

1. Create mappings from KG only:
   - `entity2id`: all unique entities in KG → `[0..N-1]`
   - `relation2id`: all unique predicates in KG → `[0..R-1]`
2. Persist:
   - `artifacts/entity2id.json`
   - `artifacts/relation2id.json`
3. Convert KG triples to integer arrays (or a dataframe):
   - `h = entity2id[subject]`
   - `r = relation2id[predicate]`
   - `t = entity2id[object]`

**Tests (before moving on):**
- Ensure mapping sizes match unique counts.
- Ensure conversion produces integer arrays in valid ranges.

### 5) Train KG embeddings (Stage A)
Implementation: `kg_project/kg_embed/pykeen_train.py` (+ optional `subgraph.py`)

Use PyKEEN.

**Defaults:**
- model: RotatE
- embedding dim: 200
- epochs: 50
- batch size: 4096 (or as large as feasible)
- optimizer: Adam, lr=1e-3

**Outputs:**
- entity embeddings → `artifacts/node_embeddings.npy`
- optionally save pykeen pipeline results/model

**If KG is too large:**
Implement optional subgraph extraction mode:
- seed nodes = entities appearing in labels (c1,c2,target)
- BFS k=2 or k=3 hops on KG
- train on induced subgraph

**Tests (before moving on):**
- Smoke test: run embedding training for 1 epoch on a tiny sampled KG and assert:
  - `node_embeddings.npy` exists
  - shape is `[num_entities, embed_dim]`
  - no NaNs in embeddings

### 6) Create train/val/test splits (Stage B)
Implementation: `kg_project/splits/splitters.py`

Implement **two split modes** and report both.

#### Split A: random (baseline)
- Stratified by label
- 80% train / 10% val / 10% test
- Save to `artifacts/splits/random/`

#### Split B: pair holdout (recommended)
- Unique pair key = `(c1,c2)`
- Split pairs into 80/10/10
- Assign rows by pair membership
- Ensure no pair overlap across splits
- Save to `artifacts/splits/pair_holdout/`

**Leakage audit logs (must print):**
- % chemicals in test seen in train
- % targets in test seen in train
- % pairs in test seen in train (should be 0 for pair holdout)

**Tests (before moving on):**
- Unit test: verify no overlap of pairs between splits in pair-holdout.
- Unit test: verify stratification roughly preserved in random split.

### 7) Feature builder + Dataset
Implementation: `kg_project/features/triple_features.py` + `kg_project/model/dataset.py`

1. Load embeddings matrix `E`
2. Convert `c1,c2,target` strings to integer ids using `entity2id`
3. Construct feature vector `x` exactly as specified (pure function)
4. Dataset returns `(x, label)`

**Tests (before moving on):**
- Unit test: feature dimension is correct.
- Unit test: swapping chemicals `(c1,c2)` does not change `x` (order invariance).

### 8) Train classifier (Stage B)
Implementation: `kg_project/model/mlp.py` + `kg_project/model/train.py`

1. Loss:
   - compute `pos_weight = n_neg / n_pos` on train
   - use `BCEWithLogitsLoss(pos_weight=...)` OR balanced sampler
2. Training defaults:
   - hidden sizes: [512, 256]
   - dropout: 0.2
   - batch size: 512
   - epochs: 30
   - early stopping on **validation ROC AUC** (patience 5)
3. Save best weights:
   - `artifacts/classifier.pt`

**Tests (before moving on):**
- `test_train_step.py`: one forward/backward step on tiny batch (assert loss finite and weights update).
- Smoke test: train for 1–2 epochs on a small subset, ensure AUC computes and no crashes.

### 9) Evaluate and write reports
Implementation: `kg_project/eval/metrics.py` + `kg_project/eval/predict.py`

On val and test:
- ROC AUC
- PR AUC
- Choose threshold by Youden J on validation ROC
- Accuracy, sensitivity, specificity
- Confusion matrix

Write:
- `artifacts/metrics.json`
- `artifacts/report.txt`
- `artifacts/predictions_test.csv` with:
  - `c1,c2,target,label,pred_prob`

Print a clean terminal summary.

---

## Required sanity checks / ablations (must implement)

### Ablation 1: ID-only baseline (no KG)
- Learn embedding tables for entities only from labeled data (random init, train end-to-end)
- Same MLP head
- Compare metrics to KG-embedding pipeline

### Ablation 2: Random embedding baseline
- Replace `E` with random Gaussian embeddings (fixed seed)
- Train classifier
- Should be near chance (if not, split leakage exists)

---

## Implementation deliverable

### `run_project.py` (thin orchestrator)
`run_project.py` should:
- parse args
- call modular functions from `kg_project/`
- avoid embedding large logic directly in the CLI

Command-line arguments:
- `--src_dir ./src`
- `--out_dir ./artifacts`
- `--kg_glob "part-*.snappy.parquet"`
- `--pos_csv gt_two_component_positive.csv`
- `--neg_csv gt_two_component_negative.csv`
- `--kg_model rotate|complex`
- `--embed_dim 200`
- `--split_mode random|pair_holdout|both`
- `--seed 42`
- `--subgraph_hops 0|2|3` (0 = full KG)
- `--run_tests 0|1` (if 1, run tests/smoke checks after each stage)

Behavior:
- If artifacts exist, allow `--force` to recompute.
- Running once should execute the full pipeline and produce all outputs.

Example:
```bash
python run_project.py \
  --src_dir ./src \
  --out_dir ./artifacts \
  --kg_model rotate \
  --embed_dim 200 \
  --split_mode both \
  --seed 42 \
  --run_tests 1
```

---

## Notes
- Treat chemical pairs as unordered everywhere.
- Avoid leakage: pair holdout is the main reported split.
- Always log counts and overlap stats.
- Prefer small, pure functions with clear inputs/outputs; write tests before scaling up.
