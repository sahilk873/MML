#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kg_project.eval.predict import write_metrics, write_predictions, write_report
from kg_project.io.csv_loader import read_label_csvs, save_labels_clean
from kg_project.io.parquet_loader import list_kg_parts, log_kg_stats, persist_kg, read_kg
from kg_project.kg_embed.pykeen_train import ensure_real_embeddings, train_pykeen
from kg_project.kg_embed.subgraph import extract_subgraph
from kg_project.model.dataset import FeatureDataset, IDTripleDataset
from kg_project.model.mlp import FeatureClassifier, IDEmbeddingModel
from kg_project.model.train import train_and_evaluate
from kg_project.preprocess.coverage_filter import filter_labels_by_entities
from kg_project.preprocess.mappings import build_id_mappings
from kg_project.preprocess.normalize_pairs import normalize_label_pairs
from kg_project.splits.splitters import (
    create_pair_holdout_splits,
    create_random_splits,
    leakage_audit,
    save_split_dfs,
)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_project")

_CPU_COUNT = os.cpu_count() or 1
DEFAULT_NUM_WORKERS = max(1, min(8, _CPU_COUNT // 2))


def format_artifact_path(path: Path, root: Path) -> str:
    """Return path relative to the run directory when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def save_classifier_artifact(
    model: nn.Module,
    mode: str,
    out_dir: Path,
    threshold: float,
    feature_input_dim: int,
    feature_embedding_dim: int,
    metrics: dict,
    split_sizes: dict,
):
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    weight_path = models_dir / f"classifier_{mode}.pt"
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(cpu_state, weight_path)
    metadata = {
        "split": mode,
        "model_class": model.__class__.__name__,
        "feature_input_dim": feature_input_dim,
        "feature_embedding_dim": feature_embedding_dim,
        "best_threshold": threshold,
        "metrics": metrics,
        "split_sizes": split_sizes,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    metadata_path = models_dir / f"classifier_{mode}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return weight_path, metadata_path


def write_run_summary(out_dir: Path, notes: list[str]) -> Path:
    summary_path = out_dir / "run_summary.txt"
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"Run completed: {timestamp}",
        f"Artifacts directory: {out_dir}",
        "",
        "Saved artifacts:",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None recorded")
    summary_path.write_text("\n".join(lines))
    return summary_path


def make_dataloader(dataset, batch_size, shuffle=False):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": True,
    }
    if DEFAULT_NUM_WORKERS > 0:
        loader_kwargs.update(
            {
                "num_workers": DEFAULT_NUM_WORKERS,
                "prefetch_factor": 2,
                "persistent_workers": True,
            }
        )
    return DataLoader(**loader_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="KG-informed two-component association pipeline")
    parser.add_argument("--src_dir", default="./src", help="Source directory housing inputs")
    parser.add_argument("--out_dir", default="./artifacts", help="Destination for pipeline outputs")
    parser.add_argument("--kg_glob", default="part-*.snappy.parquet", help="KG partition glob")
    parser.add_argument("--pos_csv", default="gt_two_component_positive.csv")
    parser.add_argument("--neg_csv", default="gt_two_component_negative.csv")
    parser.add_argument("--kg_model", choices=["rotate", "complex"], default="rotate")
    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--embed_epochs", type=int, default=50, help="PyKEEN training epochs")
    parser.add_argument("--embed_batch_size", type=int, default=4096, help="PyKEEN training batch size")
    parser.add_argument("--embed_lr", type=float, default=1e-3, help="PyKEEN optimizer learning rate")
    parser.add_argument(
        "--split_mode",
        choices=["random", "pair_holdout", "both"],
        default="both",
        help="Which split strategy to build/evaluate",
    )
    parser.add_argument("--classifier_epochs", type=int, default=30, help="MLP training epochs")
    parser.add_argument("--classifier_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--classifier_batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--subgraph_hops",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Hop-limited subgraph around label entities (0 = full KG)",
    )
    parser.add_argument(
        "--max_kg_triples",
        type=int,
        default=0,
        help="Limit the KG to the first N triples (0 = no limit)",
    )
    parser.add_argument(
        "--run_tests",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to run pytest before the pipeline",
    )
    parser.add_argument("--force", action="store_true", help="Re-run pipeline if artifacts exist")
    return parser.parse_args()


def ensure_outdir(out_dir: Path, force: bool):
    if out_dir.exists():
        if force:
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("Reusing existing artifacts at %s (use --force to rebuild)", out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)


def summarize_label_stats(df):
    examples = len(df)
    positives = int(df["label"].sum())
    negatives = examples - positives
    unique_chemicals = pd.unique(df[["c1", "c2"]].values.ravel()).size
    unique_targets = df["target"].nunique()
    return {
        "examples": examples,
        "positives": positives,
        "negatives": negatives,
        "unique_chemicals": unique_chemicals,
        "unique_targets": unique_targets,
        "conflicts_resolved": None,
    }


def ensure_labels_for_splitting(df: pd.DataFrame):
    if df.empty:
        raise RuntimeError("Coverage filtering removed all label rows; pipeline cannot proceed.")
    label_counts = df["label"].value_counts()
    if label_counts.size < 2:
        raise RuntimeError(f"Need both positive and negative labels after filtering; found {label_counts.to_dict()}.")
    return label_counts.to_dict()


def choose_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        logger.info("CUDA not available; using CPU for all models")
        return torch.device("cpu")
    try:
        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)
        name = torch.cuda.get_device_name(device_index)
    except (AssertionError, RuntimeError) as exc:
        logger.warning("Unable to inspect CUDA device (%s); falling back to CPU", exc)
        return torch.device("cpu")
    major, minor = capability
    if major + minor / 10 < 7.0:
        logger.warning(
            "CUDA device %s (sm_%d%d) is below the supported capability threshold (7.0); using CPU instead",
            name,
            major,
            minor,
        )
        return torch.device("cpu")
    logger.info("Using CUDA device %s (sm_%d%d)", name, major, minor)
    return torch.device(f"cuda:{device_index}")


def limit_kg_with_label_entities(kg_df: pd.DataFrame, labels_df: pd.DataFrame, max_triples: int, seed: int):
    if max_triples <= 0 or len(kg_df) <= max_triples:
        return kg_df, 0
    if labels_df.empty:
        logger.warning("No label rows available; falling back to random KG sampling (%d triples)", max_triples)
        limited = kg_df.sample(n=max_triples, random_state=seed).reset_index(drop=True)
        return limited, 0
    label_entities = set(pd.unique(labels_df[["c1", "c2", "target"]].values.ravel()))
    if not label_entities:
        logger.warning("Label entities set is empty; falling back to random KG sampling (%d triples)", max_triples)
        limited = kg_df.sample(n=max_triples, random_state=seed).reset_index(drop=True)
        return limited, 0
    coverage_mask = kg_df["subject"].isin(label_entities) | kg_df["object"].isin(label_entities)
    label_edges = kg_df[coverage_mask]
    label_edge_count = len(label_edges)
    if label_edge_count == 0:
        logger.warning("No KG edges matched the label entities; falling back to random KG sampling (%d triples)", max_triples)
        limited = kg_df.sample(n=max_triples, random_state=seed).reset_index(drop=True)
        return limited, 0
    if label_edge_count >= max_triples:
        limited = label_edges.sample(n=max_triples, random_state=seed).reset_index(drop=True)
        return limited, max_triples
    remaining = kg_df[~coverage_mask]
    additional_needed = max_triples - label_edge_count
    additional_count = min(additional_needed, len(remaining))
    if additional_count > 0:
        sampled_additional = remaining.sample(n=additional_count, random_state=seed)
        combined = pd.concat([label_edges, sampled_additional], ignore_index=True)
    else:
        combined = label_edges.copy()
    combined = combined.drop_duplicates()
    if len(combined) > max_triples:
        combined = combined.sample(n=max_triples, random_state=seed)
    limited = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    return limited, label_edge_count


def record_progress(out_dir: Path, stage: str, status: str, detail: str = ""):
    progress_path = out_dir / "progress.json"
    progress = {}
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
        except json.JSONDecodeError:
            progress = {}
    progress[stage] = {
        "status": status,
        "detail": detail,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    progress_path.write_text(json.dumps(progress, indent=2))
    logger.info("Progress updated: %s -> %s", stage, status)


def run_pytest():
    logger.info("Running pytest suite (-q)")
    result = subprocess.run(["python", "-m", "pytest", "-q"], check=False)
    if result.returncode == 0:
        logger.info("Pytest completed successfully")
    elif result.returncode == 5:
        logger.warning("Pytest reported no tests (exit code 5); continuing")
    else:
        logger.error("Pytest failed (exit code %d)", result.returncode)
        raise SystemExit("Pytest reported failures")


def run_ablation_baselines(pair_splits, entity2id, embeddings, device, args):
    train, val, test = pair_splits
    out = {}
    batch_size = args.classifier_batch_size
    pos_weight = (len(train) - train["label"].sum()) / max(train["label"].sum(), 1)
    pos_weight_tensor = torch.tensor(pos_weight, device=device)
    feature_dim = embeddings.shape[1]
    id_train = IDTripleDataset(train, entity2id)
    id_val = IDTripleDataset(val, entity2id)
    id_test = IDTripleDataset(test, entity2id)
    id_train_loader = make_dataloader(id_train, batch_size, shuffle=True)
    id_val_loader = make_dataloader(id_val, batch_size)
    id_test_loader = make_dataloader(id_test, batch_size)
    model = IDEmbeddingModel(len(entity2id), args.embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    results = train_and_evaluate(
        model,
        id_train_loader,
        id_val_loader,
        id_test_loader,
        criterion,
        optimizer,
        device,
        max_epochs=args.classifier_epochs,
        patience=args.classifier_patience,
    )
    out["id_only"] = {
        "val_metrics": results["val_metrics"],
        "test_metrics": results["test_metrics"],
    }
    rng = np.random.default_rng(args.seed)
    random_embeddings = rng.standard_normal(embeddings.shape).astype(np.float32)
    random_train_loader = make_dataloader(
        FeatureDataset(train, entity2id, random_embeddings),
        batch_size,
        shuffle=True,
    )
    random_val_loader = make_dataloader(FeatureDataset(val, entity2id, random_embeddings), batch_size)
    random_test_loader = make_dataloader(FeatureDataset(test, entity2id, random_embeddings), batch_size)
    random_model = FeatureClassifier(7 * feature_dim).to(device)
    random_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    random_optimizer = torch.optim.Adam(random_model.parameters(), lr=1e-3)
    random_results = train_and_evaluate(
        random_model,
        random_train_loader,
        random_val_loader,
        random_test_loader,
        random_criterion,
        random_optimizer,
        device,
        max_epochs=args.classifier_epochs,
        patience=args.classifier_patience,
    )
    out["random_embedding"] = {
        "val_metrics": random_results["val_metrics"],
        "test_metrics": random_results["test_metrics"],
    }
    return out


def main():
    args = parse_args()
    if args.run_tests:
        run_pytest()
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    ensure_outdir(out_dir, args.force)
    artifact_notes: list[str] = []
    device = choose_device()
    if args.run_tests:
        record_progress(out_dir, "tests", "complete", "pytest run before pipeline")
    kg_cache_path = out_dir / "kg_edges.parquet"
    kg_cached = kg_cache_path.exists() and not args.force
    if kg_cached:
        kg_df = pd.read_parquet(kg_cache_path)
        logger.info("Reusing cached KG edges (%d) from %s", len(kg_df), kg_cache_path)
    else:
        kg_parts = list_kg_parts(src_dir, args.kg_glob)
        kg_df = read_kg(kg_parts)
        logger.info("Loaded KG edges (%d) from %s", len(kg_df), src_dir)
        persist_kg(kg_df, out_dir)
    labels_clean_path = out_dir / "labels_clean.parquet"
    labels_cached = labels_clean_path.exists() and not args.force
    if labels_cached:
        labels_cleaned = pd.read_parquet(labels_clean_path)
        norm_stats = summarize_label_stats(labels_cleaned)
        logger.info("Reusing normalized labels (%d entries) from %s", len(labels_cleaned), labels_clean_path)
    else:
        labels_raw = read_label_csvs(src_dir, args.pos_csv, args.neg_csv)
        labels_cleaned, norm_stats = normalize_label_pairs(labels_raw)
        save_labels_clean(labels_cleaned, out_dir)
    norm_stats_log = dict(norm_stats)
    conflict_display = norm_stats.get("conflicts_resolved")
    norm_stats_log["conflicts_resolved"] = conflict_display if conflict_display is not None else "unknown"
    logger.info(
        "Labels normalized: %(examples)d examples (%(positives)d positives / %(negatives)d negatives), uniques: %(unique_chemicals)d chemicals, %(unique_targets)d targets, conflicts: %(conflicts_resolved)s",
        norm_stats_log,
    )
    record_progress(
        out_dir,
        "labels_normalized",
        "complete",
        "cached normalized labels" if labels_cached else f"{norm_stats['examples']} examples",
    )
    if args.max_kg_triples and args.max_kg_triples > 0 and len(kg_df) > args.max_kg_triples:
        kg_df, label_edge_count = limit_kg_with_label_entities(kg_df, labels_cleaned, args.max_kg_triples, args.seed)
        if label_edge_count:
            detail = (
                f"{len(kg_df)} triples with {label_edge_count} touching label entities"
            )
        else:
            detail = f"{len(kg_df)} triples"
        logger.info("Limiting KG to %s (max=%d)", detail, args.max_kg_triples)
    stats = log_kg_stats(kg_df)
    record_progress(
        out_dir,
        "kg_ingestion",
        "complete",
        "cached KG edges" if kg_cached else f"{stats['edges']} edges",
    )
    entity_set = set(pd.unique(kg_df[["subject", "object"]].values.ravel()))
    labels_filtered_path = out_dir / "labels_filtered.parquet"
    dropped_path = out_dir / "dropped_missing_entities.csv"
    coverage_cached = labels_filtered_path.exists() and dropped_path.exists() and not args.force
    if coverage_cached:
        labels_filtered = pd.read_parquet(labels_filtered_path)
        dropped = pd.read_csv(dropped_path)
        kept = len(labels_filtered)
        drop_pct = (1 - kept / len(labels_cleaned)) * 100 if len(labels_cleaned) else 0.0
        coverage_result = {
            "kept_df": labels_filtered,
            "dropped_df": dropped,
            "stats": {"kept": kept, "dropped": len(dropped), "drop_pct": drop_pct, "top_prefixes": []},
        }
    else:
        coverage_result = filter_labels_by_entities(labels_cleaned, entity_set, out_dir)
        dropped = coverage_result["dropped_df"]
    labels_filtered = coverage_result["kept_df"]
    ensure_labels_for_splitting(labels_filtered)
    logger.info(
        "Coverage filtering kept %d rows (dropped %d rows, %.2f%% drop)",
        len(labels_filtered),
        len(dropped),
        coverage_result["stats"]["drop_pct"],
    )
    record_progress(
        out_dir,
        "labels_filtered",
        "complete",
        "cached filtered labels" if coverage_cached else f"{len(labels_filtered)} rows retained",
    )
    entity2id_path = out_dir / "entity2id.json"
    relation2id_path = out_dir / "relation2id.json"
    mapping_cached = entity2id_path.exists() and relation2id_path.exists() and not args.force
    if mapping_cached:
        entity2id = json.loads(entity2id_path.read_text())
        relation2id = json.loads(relation2id_path.read_text())
    else:
        entity2id, relation2id = build_id_mappings(kg_df, out_dir)
    artifact_notes.append(f"Entity ID mapping: {format_artifact_path(entity2id_path, out_dir)}")
    artifact_notes.append(f"Relation ID mapping: {format_artifact_path(relation2id_path, out_dir)}")
    record_progress(
        out_dir,
        "mappings_built",
        "complete",
        "cached mappings" if mapping_cached else f"{len(entity2id)} entities, {len(relation2id)} relations",
    )
    seeds = set(labels_filtered[["c1", "c2", "target"]].values.ravel())
    kg_for_embeddings = extract_subgraph(kg_df, seeds, args.subgraph_hops)
    embeddings_path = out_dir / "node_embeddings.npy"
    embeddings_cached = embeddings_path.exists() and not args.force
    embedding_stats_path = out_dir / "embedding_stats.json"
    if embeddings_cached:
        embeddings = np.load(embeddings_path)
        if embedding_stats_path.exists():
            embed_stats = json.loads(embedding_stats_path.read_text())
        else:
            embed_stats = {"entities": embeddings.shape[0], "dim": embeddings.shape[1], "epochs": args.embed_epochs}
        needs_resave = np.iscomplexobj(embeddings) or embeddings.dtype != np.float32
        if needs_resave:
            embeddings = ensure_real_embeddings(embeddings)
            np.save(embeddings_path, embeddings)
        if embed_stats.get("dim") != embeddings.shape[1] or needs_resave:
            embed_stats["dim"] = embeddings.shape[1]
            embedding_stats_path.write_text(json.dumps(embed_stats, indent=2))
        logger.info("Reusing cached embeddings (%d × %d) from %s", embed_stats["entities"], embed_stats["dim"], embeddings_path)
    else:
        embeddings, embed_stats = train_pykeen(
            kg_for_embeddings,
            entity2id,
            relation2id,
            args.kg_model,
            args.embed_dim,
            epochs=args.embed_epochs,
            batch_size=args.embed_batch_size,
            lr=args.embed_lr,
            seed=args.seed,
            out_dir=out_dir,
            device=device,
        )
        embedding_stats_path.write_text(json.dumps(embed_stats, indent=2))
    artifact_notes.append(f"Node embeddings: {format_artifact_path(embeddings_path, out_dir)}")
    artifact_notes.append(f"Embedding stats: {format_artifact_path(embedding_stats_path, out_dir)}")
    record_progress(
        out_dir,
        "kg_embeddings",
        "complete",
        "cached embeddings" if embeddings_cached else f"{embed_stats['entities']} vectors × {embed_stats['dim']} dim; epochs {embed_stats['epochs']}",
    )
    feature_embedding_dim = embeddings.shape[1]
    feature_input_dim = 7 * feature_embedding_dim
    split_modes = [args.split_mode] if args.split_mode != "both" else ["random", "pair_holdout"]
    metrics = {}
    main_predictions = None
    chosen_test_df = None
    pair_holdout_splits = None
    for mode in split_modes:
        if mode == "random":
            train_df, val_df, test_df = create_random_splits(labels_filtered, args.seed)
        else:
            train_df, val_df, test_df = create_pair_holdout_splits(labels_filtered, args.seed)
            pair_holdout_splits = (train_df, val_df, test_df)
        logger.info(
            "%s split sizes: train=%d val=%d test=%d",
            mode,
            len(train_df),
            len(val_df),
            len(test_df),
        )
        save_split_dfs(mode, train_df, val_df, test_df, out_dir)
        split_base = out_dir / "splits" / mode
        artifact_notes.append(
            f"{mode} split parquet files: {format_artifact_path(split_base, out_dir)}"
        )
        leakage = leakage_audit(train_df, test_df, mode)
        train_dataset = FeatureDataset(train_df, entity2id, embeddings)
        val_dataset = FeatureDataset(val_df, entity2id, embeddings)
        test_dataset = FeatureDataset(test_df, entity2id, embeddings)
        batch_size = args.classifier_batch_size
        train_loader = make_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = make_dataloader(val_dataset, batch_size)
        test_loader = make_dataloader(test_dataset, batch_size)
        model = FeatureClassifier(feature_input_dim).to(device)
        pos_weight = (len(train_df) - train_df["label"].sum()) / max(train_df["label"].sum(), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logger.info(
            "Training classifier for %s split (batch=%d, epochs=%d, patience=%d)",
            mode,
            batch_size,
            args.classifier_epochs,
            args.classifier_patience,
        )
        results = train_and_evaluate(
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            max_epochs=args.classifier_epochs,
            patience=args.classifier_patience,
        )
        val_roc = results["val_metrics"].get("roc_auc")
        test_roc = results["test_metrics"].get("roc_auc")
        logger.info(
            "%s split training complete: val ROC=%.4f test ROC=%.4f threshold=%.4f",
            mode,
            val_roc if val_roc is not None else float("nan"),
            test_roc if test_roc is not None else float("nan"),
            results["best_threshold"],
        )
        metrics[mode] = {
            "val_metrics": results["val_metrics"],
            "test_metrics": results["test_metrics"],
            "leakage": leakage,
            "threshold": results["best_threshold"],
        }
        val_roc = metrics[mode]["val_metrics"].get("roc_auc")
        detail = f"val ROC {val_roc:.4f}" if val_roc is not None else "val ROC N/A"
        record_progress(out_dir, f"classifier_{mode}", "complete", detail)
        split_sizes = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
        weight_path, metadata_path = save_classifier_artifact(
            model,
            mode,
            out_dir,
            results["best_threshold"],
            feature_input_dim,
            feature_embedding_dim,
            metrics[mode],
            split_sizes,
        )
        artifact_notes.append(
            f"{mode} classifier weights: {format_artifact_path(weight_path, out_dir)} (metadata: {format_artifact_path(metadata_path, out_dir)})"
        )
        if mode == "pair_holdout" or (mode == "random" and not main_predictions):
            main_predictions = (test_df, results["test_probs"], mode)
            chosen_test_df = test_df
    if pair_holdout_splits:
        metrics["ablations"] = run_ablation_baselines(pair_holdout_splits, entity2id, embeddings, device, args)
        record_progress(out_dir, "ablations", "complete", "ID-only & random embedding baselines")
    else:
        metrics["ablations"] = {}
    write_metrics(metrics, out_dir)
    artifact_notes.append(f"Metrics JSON: {format_artifact_path(out_dir / 'metrics.json', out_dir)}")
    report_lines = [
        f"KG edges: {stats['edges']}",
        f"Clean labels: {norm_stats['examples']} examples",
        f"Filtered labels: {len(labels_filtered)} examples",
    ]
    for mode in split_modes:
        if mode in metrics:
            report_lines.append(f"{mode} test ROC AUC = {metrics[mode]['test_metrics']['roc_auc']:.4f}")
    if metrics.get("ablations"):
        report_lines.append("Ablation (ID-only) ROC AUC = {:.4f}".format(metrics["ablations"]["id_only"]["test_metrics"]["roc_auc"]))
        report_lines.append("Ablation (random embeddings) ROC AUC = {:.4f}".format(metrics["ablations"]["random_embedding"]["test_metrics"]["roc_auc"]))
    write_report(report_lines, out_dir)
    artifact_notes.append(f"Report: {format_artifact_path(out_dir / 'report.txt', out_dir)}")
    if main_predictions and chosen_test_df is not None:
        write_predictions(chosen_test_df, main_predictions[1], out_dir, main_predictions[2])
        artifact_notes.append(
            f"Predictions CSV: {format_artifact_path(out_dir / 'predictions_test.csv', out_dir)}"
        )
    record_progress(out_dir, "evaluation", "complete", "Metrics, report, predictions written")
    summary_path = write_run_summary(out_dir, artifact_notes)
    logger.info(
        "Pipeline completed; outputs under %s (summary: %s)",
        out_dir,
        format_artifact_path(summary_path, out_dir),
    )


if __name__ == "__main__":
    main()



#sample prompt
#python run_project.py --src_dir ./src --out_dir ./artifacts --kg_model rotate --embed_dim 200 --embed_epochs 1 --embed_batch_size 512 --embed_lr 1e-3 --classifier_epochs 5 --classifier_batch_size 128 --split_mode both --seed 42 --subgraph_hops 0 --run_tests 1
#python run_project.py --src_dir ./src --out_dir ./artifacts --kg_model rotate --embed_dim 200 --embed_epochs 50 --embed_batch_size 4096 --embed_lr 1e-3 --classifier_epochs 30 --classifier_patience 5 --classifier_batch_size 512 --split_mode both --seed 42 --subgraph_hops 2 --run_tests 0 --force


'''python run_project.py \
    --max_kg_triples 50000 \
    --embed_epochs 1 \
    --embed_batch_size 256 \
    --classifier_epochs 1 \
    --classifier_batch_size 128 \
    --force
'''
