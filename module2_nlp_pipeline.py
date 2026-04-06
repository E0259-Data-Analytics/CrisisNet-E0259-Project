from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DATA_ROOT = Path("crisisnet-data")


@dataclass
class TenKRecord:
    ticker: str
    filing_date: pd.Timestamp
    year: int
    item_1a: str
    item_7: str


def quarter_from_date(dt: pd.Timestamp) -> str:
    q = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{q}"

def quarter_start_date(q: str) -> pd.Timestamp:
    year = int(q[:4])
    quarter = int(q[-1])
    month = 1 + (quarter - 1) * 3
    return pd.Timestamp(year=year, month=month, day=1)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = " ".join(text.replace("\n", " ").replace("\t", " ").split())
    return cleaned.lower()


def load_ticker_cik_map(map_path: Path) -> Dict[int, str]:
    df = pd.read_csv(map_path)
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce")
    mapping = {}
    for _, row in df.iterrows():
        if pd.notna(row["cik"]) and isinstance(row["ticker"], str):
            mapping[int(row["cik"])] = row["ticker"]
    return mapping


def load_filings_metadata(meta_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    meta["CIK"] = pd.to_numeric(meta["CIK"], errors="coerce")
    meta["filing_date"] = pd.to_datetime(meta["Filing Date"], errors="coerce")
    return meta


def iter_10k_records(
    tenk_dir: Path,
    meta: pd.DataFrame,
    cik_to_ticker: Dict[int, str],
) -> Iterator[TenKRecord]:
    for fp in tenk_dir.glob("*.json"):
        parts = fp.stem.split("_")
        if len(parts) < 3:
            continue
        cik = pd.to_numeric(parts[0], errors="coerce")
        year = pd.to_numeric(parts[2], errors="coerce")
        if pd.isna(cik) or pd.isna(year):
            continue
        cik = int(cik)
        year = int(year)
        ticker = cik_to_ticker.get(cik)
        if not ticker:
            continue

        meta_subset = meta[(meta["CIK"] == cik) & (meta["filing_date"].dt.year == year)]
        filing_date = meta_subset["filing_date"].min()
        if pd.isna(filing_date):
            continue

        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        item_1a = payload.get("item_1A") or payload.get("item_1a") or ""
        item_7 = payload.get("item_7") or ""

        yield TenKRecord(
            ticker=ticker,
            filing_date=pd.to_datetime(filing_date),
            year=year,
            item_1a=clean_text(item_1a),
            item_7=clean_text(item_7),
        )


def fit_lda_topic_model(
    texts: Sequence[str],
    n_topics: int = 12,
    max_features: int = 6000,
    random_state: int = 42,
):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
    )
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=20,
    )
    lda.fit(dtm)
    return lda, vectorizer


def topic_proportions(lda, vectorizer, texts: Sequence[str]) -> np.ndarray:
    dtm = vectorizer.transform(texts)
    return lda.transform(dtm)


class LexiconSentiment:
    pos_words = {
        "growth", "improve", "improved", "strong", "positive", "profit", "profitable",
        "increase", "increased", "record", "confidence", "optimistic", "opportunity",
    }
    neg_words = {
        "bankrupt", "bankruptcy", "liquidity", "default", "covenant", "distress",
        "impairment", "restructuring", "material weakness", "going concern",
        "adverse", "decline", "deteriorate", "loss", "losses", "liability",
        "litigation", "risk", "uncertain", "uncertainty", "negative",
    }

    def score(self, text: str) -> Tuple[float, float, float, float]:
        words = text.split()
        if not words:
            return 0.0, 0.0, 1.0, 0.0
        pos = sum(1 for w in words if w in self.pos_words)
        neg = sum(1 for w in words if w in self.neg_words)
        total = len(words)
        pos_r = pos / total
        neg_r = neg / total
        neu_r = max(0.0, 1.0 - pos_r - neg_r)
        score = pos_r - neg_r
        return pos_r, neg_r, neu_r, score


class LexiconUncertainty:
    def __init__(self, lexicon_path: Optional[Path] = None):
        default_terms = {
            "uncertain", "uncertainty", "unpredictable", "indefinite", "ambiguous",
            "contingent", "volatile", "pending", "fluctuate", "hesitant", "approximately",
            "may", "might", "could", "possibly", "risk", "subject",
        }
        self.phrases = {"subject to", "going concern", "material weakness"}
        self.terms = set(default_terms)
        if lexicon_path and lexicon_path.exists():
            terms = set()
            for line in lexicon_path.read_text(encoding="utf-8").splitlines():
                t = line.strip().lower()
                if not t:
                    continue
                if " " in t:
                    self.phrases.add(t)
                else:
                    terms.add(t)
            if terms:
                self.terms = terms

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        text = clean_text(text)
        words = text.split()
        if not words:
            return 0.0
        count = sum(1 for w in words if w in self.terms)
        for phrase in self.phrases:
            count += text.count(phrase)
        return float(count / max(1, len(words)))


class FinBertSentiment:
    def __init__(self, model_name: str, device: int = -1):
        from transformers import pipeline

        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
        )

    @staticmethod
    def _sent_tokenize(text: str) -> List[str]:
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                raise RuntimeError("nltk punkt not available")
            return nltk.sent_tokenize(text)
        except Exception:
            text = text.replace("?", ".").replace("!", ".")
            return [s.strip() for s in text.split(".") if s.strip()]

    def score(self, text: str) -> Tuple[float, float, float, float]:
        if not text:
            return 0.0, 0.0, 1.0, 0.0
        sentences = [s for s in self._sent_tokenize(text) if len(s.split()) > 8]
        if not sentences:
            return 0.0, 0.0, 1.0, 0.0

        def to_scores(output) -> Tuple[float, float, float, float]:
            label = output["label"].lower()
            score = float(output["score"])
            if "positive" in label:
                return score, 0.0, 1.0 - score, score
            if "negative" in label:
                return 0.0, score, 1.0 - score, -score
            return 0.0, 0.0, 1.0, 0.0

        batch_size = 32
        scores = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            outputs = self.pipe(batch, truncation=True)
            scores.extend(to_scores(o) for o in outputs)

        pos = float(np.mean([s[0] for s in scores]))
        neg = float(np.mean([s[1] for s in scores]))
        neu = float(np.mean([s[2] for s in scores]))
        score = float(np.mean([s[3] for s in scores]))
        return pos, neg, neu, score


def build_10k_features(
    tenk_records: Sequence[TenKRecord],
    lda,
    vectorizer,
) -> pd.DataFrame:
    item7_texts = [r.item_7 for r in tenk_records]
    topic_mat = topic_proportions(lda, vectorizer, item7_texts)
    topic_cols = [f"topic_{i}" for i in range(topic_mat.shape[1])]

    rows = []
    for rec, topic_vec in zip(tenk_records, topic_mat):
        row = {
            "ticker": rec.ticker,
            "quarter": quarter_from_date(rec.filing_date),
            **{c: float(v) for c, v in zip(topic_cols, topic_vec)},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def aggregate_sentiment(
    records: Iterable[Tuple[str, pd.Timestamp, str]],
    scorer,
) -> pd.DataFrame:
    rows = []
    for ticker, dt, text in records:
        text = clean_text(text)
        pos, neg, neu, score = scorer.score(text)
        rows.append({
            "ticker": ticker,
            "quarter": quarter_from_date(dt),
            "pos": pos,
            "neg": neg,
            "neu": neu,
            "score": score,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "quarter", "pos", "neg", "neu", "score"])
    agg = df.groupby(["ticker", "quarter"], as_index=False).mean()
    return agg


def aggregate_uncertainty(
    records: Iterable[Tuple[str, pd.Timestamp, str]],
    scorer: LexiconUncertainty,
) -> pd.DataFrame:
    rows = []
    for ticker, dt, text in records:
        text = clean_text(text)
        rate = scorer.score(text)
        rows.append({
            "ticker": ticker,
            "quarter": quarter_from_date(dt),
            "uncertainty_rate": rate,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "quarter", "uncertainty_rate"])
    agg = df.groupby(["ticker", "quarter"], as_index=False).mean()
    return agg


def iter_earnings_calls(
    jsonl_path: Path,
    tickers: Optional[set] = None,
    max_records: Optional[int] = None,
) -> Iterator[Tuple[str, pd.Timestamp, str]]:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_records is not None and count >= max_records:
                break
            try:
                payload = json.loads(line)
            except Exception:
                continue
            ticker = payload.get("ticker")
            date = payload.get("date")
            if not ticker or not date:
                continue
            if tickers is not None and ticker not in tickers:
                continue
            if isinstance(date, str) and date.strip().endswith("ET"):
                date = date.replace("ET", "").strip().rstrip(",")
            try:
                dt = pd.to_datetime(date, errors="coerce")
            except Exception:
                continue
            if pd.isna(dt):
                continue
            q = payload.get("question", "")
            a = payload.get("answer", "")
            text = f"{q} {a}".strip()
            if not text:
                continue
            count += 1
            yield ticker, dt, text


def build_features(
    n_topics: int = 12,
    max_features: int = 6000,
    random_state: int = 42,
    finbert_model: Optional[str] = None,
    finbert_device: Optional[int] = -1,
    max_transcripts: Optional[int] = None,
    return_artifacts: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    tenk_dir = DATA_ROOT / "Module_2" / "10k_extracted" / "10-K"
    meta_path = DATA_ROOT / "Module_2" / "filings_metadata.csv"
    map_path = DATA_ROOT / "Module_1" / "sec_xbrl" / "ticker_cik_mapping.csv"
    transcripts_path = DATA_ROOT / "Module_2" / "transcripts" / "huggingface" / "train.jsonl"

    meta = load_filings_metadata(meta_path)
    cik_to_ticker = load_ticker_cik_map(map_path)
    tenk_records = list(iter_10k_records(tenk_dir, meta, cik_to_ticker))
    if not tenk_records:
        raise RuntimeError("No 10-K records found. Check input paths.")

    item7_texts = [r.item_7 for r in tenk_records]
    lda, vectorizer = fit_lda_topic_model(
        item7_texts,
        n_topics=n_topics,
        max_features=max_features,
        random_state=random_state,
    )
    topic_df = build_10k_features(tenk_records, lda, vectorizer)

    if finbert_model:
        device = finbert_device
        if device is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except Exception:
                device = -1
        scorer = FinBertSentiment(finbert_model, device=device)
    else:
        scorer = LexiconSentiment()

    uncertainty_lexicon_path = DATA_ROOT / "Module_2" / "lexicons" / "lm_uncertainty.txt"
    uncertainty_scorer = LexiconUncertainty(uncertainty_lexicon_path)

    risk_records = [(r.ticker, r.filing_date, r.item_1a) for r in tenk_records]
    tenk_sent = aggregate_sentiment(risk_records, scorer)
    tenk_sent = tenk_sent.add_prefix("tenk_")
    tenk_sent = tenk_sent.rename(columns={"tenk_ticker": "ticker", "tenk_quarter": "quarter"})
    tenk_unc = aggregate_uncertainty(risk_records, uncertainty_scorer)
    tenk_unc = tenk_unc.add_prefix("tenk_")
    tenk_unc = tenk_unc.rename(columns={"tenk_ticker": "ticker", "tenk_quarter": "quarter"})

    tickers = {r.ticker for r in tenk_records}
    calls = iter_earnings_calls(transcripts_path, tickers=tickers, max_records=max_transcripts)
    calls_sent = aggregate_sentiment(calls, scorer)
    calls_sent = calls_sent.add_prefix("calls_")
    calls_sent = calls_sent.rename(columns={"calls_ticker": "ticker", "calls_quarter": "quarter"})
    calls_unc = aggregate_uncertainty(calls, uncertainty_scorer)
    calls_unc = calls_unc.add_prefix("calls_")
    calls_unc = calls_unc.rename(columns={"calls_ticker": "ticker", "calls_quarter": "quarter"})

    df = topic_df.merge(tenk_sent, on=["ticker", "quarter"], how="left")
    df = df.merge(tenk_unc, on=["ticker", "quarter"], how="left")
    df = df.merge(calls_sent, on=["ticker", "quarter"], how="left")
    df = df.merge(calls_unc, on=["ticker", "quarter"], how="left")

    df["quarter_start"] = df["quarter"].map(quarter_start_date)
    df = df.sort_values(["ticker", "quarter_start"])

    if "tenk_score" in df.columns:
        df["tenk_score_4q_mean"] = (
            df.groupby("ticker")["tenk_score"]
            .transform(lambda s: s.rolling(4, min_periods=1).mean())
        )
        df["tenk_score_delta"] = df.groupby("ticker")["tenk_score"].diff()
    if "calls_score" in df.columns:
        df["calls_score_4q_mean"] = (
            df.groupby("ticker")["calls_score"]
            .transform(lambda s: s.rolling(4, min_periods=1).mean())
        )
        df["calls_score_delta"] = df.groupby("ticker")["calls_score"].diff()
    if "tenk_uncertainty_rate" in df.columns:
        df["tenk_uncertainty_4q_mean"] = (
            df.groupby("ticker")["tenk_uncertainty_rate"]
            .transform(lambda s: s.rolling(4, min_periods=1).mean())
        )
    if "calls_uncertainty_rate" in df.columns:
        df["calls_uncertainty_4q_mean"] = (
            df.groupby("ticker")["calls_uncertainty_rate"]
            .transform(lambda s: s.rolling(4, min_periods=1).mean())
        )

    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if topic_cols:
        eps = 1e-8
        def _kl_shift(group: pd.DataFrame) -> pd.Series:
            mat = group[topic_cols].to_numpy(dtype=float)
            mat = mat / np.clip(mat.sum(axis=1, keepdims=True), eps, None)
            kl = [np.nan] * len(group)
            for i in range(1, len(mat)):
                p = mat[i]
                q = mat[i - 1]
                kl[i] = float(np.sum(p * np.log((p + eps) / (q + eps))))
            return pd.Series(kl, index=group.index)

        df["topic_kl_shift"] = df.groupby("ticker", group_keys=False).apply(_kl_shift)

    df = df.drop(columns=["quarter_start"])

    if return_artifacts:
        artifacts = {
            "lda_model": lda,
            "lda_vectorizer": vectorizer,
            "n_topics": n_topics,
            "max_features": max_features,
            "random_state": random_state,
        }
        return df, artifacts
    return df
