from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module2_nlp_pipeline import DATA_ROOT, build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CrisisNet Module 2 NLP features.")
    parser.add_argument("--n-topics", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--finbert-model", type=str, default=None)
    parser.add_argument("--finbert-device", type=int, default=-1)
    parser.add_argument("--auto-device", action="store_true", help="Auto-select GPU if available (FinBERT only).")
    parser.add_argument("--max-transcripts", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_ROOT / "Module_2" / "features" / "X_nlp.parquet"),
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save LDA model and vectorizer to repo root.",
    )
    args = parser.parse_args()

    finbert_device = args.finbert_device
    if args.auto_device and args.finbert_model:
        finbert_device = None

    result = build_features(
        n_topics=args.n_topics,
        max_features=args.max_features,
        random_state=args.random_state,
        finbert_model=args.finbert_model,
        finbert_device=finbert_device,
        max_transcripts=args.max_transcripts,
        return_artifacts=args.save_artifacts,
    )
    if args.save_artifacts:
        df, artifacts = result
    else:
        df = result

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    if args.save_artifacts:
        import joblib
        lda_path = ROOT / "lda_model_module2.joblib"
        vec_path = ROOT / "lda_vectorizer_module2.joblib"
        joblib.dump(artifacts["lda_model"], lda_path)
        joblib.dump(artifacts["lda_vectorizer"], vec_path)
        meta = {
            "n_topics": artifacts["n_topics"],
            "max_features": artifacts["max_features"],
            "random_state": artifacts["random_state"],
            "finbert_model": args.finbert_model,
            "finbert_device": finbert_device,
            "max_transcripts": args.max_transcripts,
        }
        (ROOT / "module2_artifacts_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Saved LDA model to {lda_path}")
        print(f"Saved vectorizer to {vec_path}")
        print(f"Saved metadata to {ROOT / 'module2_artifacts_meta.json'}")


if __name__ == "__main__":
    main()
