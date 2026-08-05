"""Run Phase 3A daily research from local Parquet files only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from event_engine import cluster_daily_events, detect_daily_events
from event_study import select_control_days, summarize_event_study
from influence_features import build_feature_panel
from influence_report import build_phase3a_report
from influence_statistics import (apply_fdr_families, build_relationships,
                                  run_granger_all_directions)
from market_data_api import load_aligned_market_data
from predictive_influence import run_all_pairs_walk_forward
from residual_returns import build_residual_returns
from response_engine import measure_daily_responses

NY = ZoneInfo("America/New_York")
BASE = Path(__file__).resolve().parent


def load_influence_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_daily_frames(frames: dict[str, pd.DataFrame], now=None):
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(NY)
    if local.hour >= 16:
        return frames
    return {symbol: frame.loc[frame.index.tz_convert(NY).date < local.date()].copy()
            for symbol, frame in frames.items()}


def _manifest(config_path: Path, config: dict, frames: dict[str, pd.DataFrame]) -> dict:
    config_bytes = config_path.read_bytes()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE,
            check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "uncommitted-workspace"
    dependencies = {}
    for package in ("pandas", "numpy", "scipy", "pyarrow", "scikit-learn", "statsmodels"):
        try: dependencies[package] = version(package)
        except PackageNotFoundError: dependencies[package] = None
    data_root = BASE / "data"
    return {"created_at": datetime.now(timezone.utc).isoformat(),
        "engine_version": config["engine_version"], "git_commit": commit,
        "configuration_hash": hashlib.sha256(config_bytes).hexdigest(),
        "python": platform.python_version(), "dependencies": dependencies,
        "input_hashes": {symbol: _hash_file(data_root / "market" / "daily" / f"{symbol}.parquet")
                         for symbol in frames},
        "analysis_scope": {"research_symbols": config["research_symbols"],
                           "control_symbols": config["control_symbols"]},
        "input_policy": "local Parquet only; no Yahoo calls"}


def run_foundation_through_responses(config_path: Path, start=None, end=None) -> dict:
    config = load_influence_config(config_path)
    all_symbols = list(dict.fromkeys(config["research_symbols"] + config["control_symbols"]))
    frames = load_aligned_market_data(all_symbols, "1d", start, end)
    frames = _completed_daily_frames(frames)
    features = build_feature_panel(frames, int(config["feature_window"]))
    residuals = build_residual_returns(features, config["market_benchmark"],
        config["sector_benchmark"], int(config["residual_window"]),
        int(config["minimum_residual_observations"]))
    event_residuals = residuals.loc[residuals.ticker.isin(config["research_symbols"])]
    events = detect_daily_events(features, event_residuals, config)
    trading_dates = sorted(features.loc[features.ticker == config["market_benchmark"], "timestamp"])
    clusters, representatives = cluster_daily_events(events, trading_dates,
        int(config["events"]["cooldown_trading_days"]))
    responses = measure_daily_responses(representatives, residuals,
        config["research_symbols"], [int(x) for x in config["horizons"]])

    root = BASE / config["output_root"]
    paths = {name: root / folder / filename for name, folder, filename in [
        ("residuals", "residuals", "daily_residual_returns.parquet"),
        ("events", "events", "daily_events.parquet"),
        ("clusters", "events", "daily_event_clusters.parquet"),
        ("representatives", "events", "daily_representative_events.parquet"),
        ("responses", "responses", "daily_responses.parquet")]}
    values = {"residuals": residuals, "events": events, "clusters": clusters,
              "representatives": representatives, "responses": responses}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        values[name].to_parquet(path, index=False, compression="zstd")
    manifest = _manifest(config_path, config, frames)
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = build_interim_report(root, config, values)
    return {**values, "summary": summary, "manifest": manifest}


def build_interim_report(root: Path, config: dict, values: dict) -> dict:
    residuals, events, clusters, responses = (values[x] for x in
        ("residuals", "events", "clusters", "responses"))
    response_counts = responses.groupby(["horizon", "response_status"]).size().unstack(fill_value=0)
    summary = {"residual_rows": int((residuals.residual_status == "ok").sum()),
        "residuals_rejected_insufficient_history": int(
            (residuals.residual_status != "ok").sum()),
        "raw_events": len(events), "event_clusters": len(clusters),
        "representative_events": len(values["representatives"]),
        "responses_ok": int((responses.response_status == "ok").sum()),
        "responses_rejected": int((responses.response_status != "ok").sum()),
        "responses_by_horizon": {str(h): {str(k): int(v) for k, v in row.items()}
                                 for h, row in response_counts.iterrows()}}
    reports = root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    table_lines = ["| horizon | " + " | ".join(map(str, response_counts.columns)) + " |",
                   "|---:" + "|---:" * len(response_counts.columns) + "|"]
    table_lines += [f"| {h} | " + " | ".join(str(int(v)) for v in row) + " |"
                    for h, row in response_counts.iterrows()]
    text = ["# Phase 3A Interim — Foundation through Responses", "",
        "Local Parquet only. Daily events are available after close; responses begin next session.", "",
        *(f"- {key}: {value}" for key, value in summary.items() if key != "responses_by_horizon"),
        "", "## Responses by horizon", "", *table_lines]
    (reports / "phase3a_interim.md").write_text("\n".join(text), encoding="utf-8")
    (reports / "phase3a_interim.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_full_research(config_path: Path, start=None, end=None, resume=False) -> dict:
    config = load_influence_config(config_path); root = BASE / config["output_root"]
    first_paths = {"residuals": root/"residuals/daily_residual_returns.parquet",
        "events": root/"events/daily_events.parquet",
        "clusters": root/"events/daily_event_clusters.parquet",
        "representatives": root/"events/daily_representative_events.parquet",
        "responses": root/"responses/daily_responses.parquet"}
    if resume and all(path.exists() for path in first_paths.values()):
        first = {name: pd.read_parquet(path) for name,path in first_paths.items()}
    else:
        first = run_foundation_through_responses(config_path, start, end)
    residuals, representatives, responses = (first[x] for x in
        ("residuals", "representatives", "responses"))
    controls_path = root/"responses/event_study_controls.parquet"
    controls = pd.read_parquet(controls_path) if resume and controls_path.exists() else pd.DataFrame()
    if controls.empty or "event_id" not in controls:
        controls = select_control_days(representatives, responses, residuals,
            int(config.get("control_matches_per_event", 5)))
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    controls.to_parquet(controls_path, index=False, compression="zstd")
    event_summary = summarize_event_study(representatives, responses, controls,
        int(config["bootstrap_iterations"]), int(config["block_bootstrap_length"]),
        int(config["random_seed"]))
    if not event_summary.empty:
        event_summary["response_metric"] = "target_residual_return"
        event_summary["analysis_window"] = "full_phase3a_history"
        event_summary = apply_fdr_families(event_summary, "permutation_p_value",
            ["event_type", "horizon", "response_metric", "analysis_window"],
            float(config["fdr_alpha"]), "adjusted_p_value")
    event_summary.to_parquet(root/"responses/event_study_summary.parquet", index=False)

    influence = root/"influence"; influence.mkdir(parents=True, exist_ok=True)
    folds_path = influence/"predictive_model_folds.parquet"
    folds = pd.read_parquet(folds_path) if resume and folds_path.exists() else \
        run_all_pairs_walk_forward(residuals, representatives,
                                   config["research_symbols"], config["walk_forward"])
    folds.to_parquet(folds_path, index=False, compression="zstd")
    granger = run_granger_all_directions(residuals, config["research_symbols"],
                                         config["granger_lags"], config["fdr_alpha"])
    granger.to_parquet(influence/"granger_results.parquet", index=False, compression="zstd")
    relationships = build_relationships(event_summary, folds, granger, config)
    relationships.to_parquet(influence/"influence_relationships.parquet", index=False,
                             compression="zstd")
    build_phase3a_report(root, config, residuals, representatives, responses,
                         folds, granger, relationships)
    return {**first, "controls": controls, "event_summary": event_summary,
            "folds": folds, "granger": granger, "relationships": relationships}


def rebuild_reports(config_path: Path) -> None:
    config=load_influence_config(config_path); root=BASE/config["output_root"]
    read=lambda p: pd.read_parquet(root/p)
    build_phase3a_report(root, config, read("residuals/daily_residual_returns.parquet"),
        read("events/daily_representative_events.parquet"), read("responses/daily_responses.parquet"),
        read("influence/predictive_model_folds.parquet"), read("influence/granger_results.parquet"),
        read("influence/influence_relationships.parquet"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE / "influence_config.json")
    parser.add_argument("--start"); parser.add_argument("--end")
    parser.add_argument("--symbols", help="Reserved for configured research subset overrides")
    parser.add_argument("--residual-window", type=int)
    parser.add_argument("--event-types"); parser.add_argument("--horizons")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = load_influence_config(args.config)
    if args.symbols:
        requested = [x.strip().upper() for x in args.symbols.split(",")]
        allowed = set(config["research_symbols"])
        if not set(requested) <= allowed:
            parser.error("--symbols is limited to configured Phase 3A research symbols")
        config["research_symbols"] = requested
        temporary = BASE / "research" / ".runtime_config.json"
        temporary.parent.mkdir(exist_ok=True); temporary.write_text(json.dumps(config))
        args.config = temporary
    if args.residual_window:
        config["residual_window"] = args.residual_window
    if args.horizons:
        config["horizons"] = [int(x) for x in args.horizons.split(",")]
    if args.residual_window or args.horizons:
        temporary = BASE / "research" / ".runtime_config.json"
        temporary.parent.mkdir(exist_ok=True); temporary.write_text(json.dumps(config))
        args.config = temporary
    if args.report_only:
        rebuild_reports(args.config); print("Reports rebuilt from saved research outputs"); return 0
    result = run_full_research(args.config, args.start, args.end, args.resume)
    print(json.dumps({"relationships": len(result["relationships"]),
        "status": result["relationships"].relationship_status.value_counts().to_dict(),
        "folds": len(result["folds"]), "granger_tests": len(result["granger"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
