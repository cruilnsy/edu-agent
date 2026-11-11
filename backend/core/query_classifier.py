# backend/core/query_classifier.py
# Binary query classifier for the QA agent: general (LLM answers directly) or specialized
# (RAG over the knowledge base). Labels follow the knowledge base topics; see
# scripts/generate_classifier_data.py for how the training data defines them.

import json
import random
from pathlib import Path
from typing import Optional

import torch

from backend.config import get_settings, resolve_path
from backend.core.logger import get_logger

logger = get_logger(__name__)

LABEL2ID = {"general": 0, "specialized": 1}
ID2LABEL = {0: "general", 1: "specialized"}

# High bar for "general": sending an in-topic question to the bare LLM is worse than one extra
# RAG call, because the answer may contradict the knowledge base.
GENERAL_CONFIDENCE_THRESHOLD = 0.85


class QueryClassifier:
    """general / specialized query classifier.

    Training:  QueryClassifier(base_model_path).train(data_path, output_dir)
    Inference: QueryClassifier().classify("What is a React hook?")  # loads the fine-tuned model
    """

    _instance: Optional["QueryClassifier"] = None

    def __init__(self, model_path: Optional[str] = None):
        """model_path: a base or fine-tuned model to load; None loads the fine-tuned model from settings."""
        settings = get_settings()
        if model_path:
            model_id = model_path
            self._is_finetuned = model_path != resolve_path(settings.classifier_model_path)
        else:
            model_id = resolve_path(settings.finetuned_classifier_path)
            self._is_finetuned = True
        from transformers import pipeline as hf_pipeline
        self._pipeline = hf_pipeline(
            task="text-classification",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,
            truncation=True,
            max_length=128,
        )
        logger.info("query_classifier.loaded", model_id=model_id, finetuned=self._is_finetuned)

    @classmethod
    def get_instance(cls) -> "QueryClassifier":
        """Lazily created singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Training ──

    def train(
        self,
        data_path: str,
        output_dir: str,
        test_path: str,
        epochs: int = 8,
        batch_size: int = 64,
        lr: float = 5e-5,
        max_length: int = 128,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """Fine-tune the loaded base model on JSONL rows {"text", "label"} and save it to output_dir.
        Rows in test_path are a fixed held-out set: they are removed from the training data, so
        regenerating or extending the data never leaks them into training. Returns the test metrics."""
        # Training-only dependencies stay out of the inference path
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        set_seed(seed)
        test_rows = self.load_jsonl(test_path)
        held_out = {r["text"].lower() for r in test_rows}
        rows = [r for r in self.load_jsonl(data_path) if r["text"].lower() not in held_out]
        train_rows, val_rows = self.stratified_split(rows, val_ratio, seed)
        logger.info("query_classifier.split", train=len(train_rows), val=len(val_rows), test=len(test_rows))

        model_id = self._pipeline.model.config._name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,   # The 2-class head starts from random weights
        )

        def to_dataset(rows_: list[dict]) -> Dataset:
            ds = Dataset.from_dict({"text": [r["text"] for r in rows_],
                                    "label": [LABEL2ID[r["label"]] for r in rows_]})
            return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                          batched=True, remove_columns=["text"])

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            return {"accuracy": float(accuracy_score(labels, preds)), "f1_macro": float(f1)}

        use_cuda = torch.cuda.is_available()
        train_args = TrainingArguments(
            output_dir=str(Path(output_dir) / "_checkpoints"),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_total_limit=1,
            logging_steps=20,
            fp16=use_cuda,
            report_to="none",
            seed=seed,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=to_dataset(train_rows),
            eval_dataset=to_dataset(val_rows),
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        test_metrics = trainer.evaluate(to_dataset(test_rows))
        logger.info("query_classifier.test_metrics",
                    **{k: round(v, 4) for k, v in test_metrics.items() if k.startswith("eval_")})

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("query_classifier.saved", output_dir=output_dir)
        return test_metrics

    # ── Inference ──

    def classify(self, text: str) -> tuple[str, float]:
        """Return (label, confidence): "general" only when P(general) >= GENERAL_CONFIDENCE_THRESHOLD,
        otherwise ("specialized", 1 - P(general)). Unknown label names fall back to ("specialized", 0.5)."""
        raw_outputs: list[dict] = self._pipeline(text)[0]
        general_score: Optional[float] = None
        for item in raw_outputs:
            if item["label"].lower() in ("general", "label_0"):
                general_score = item["score"]
                break

        if general_score is None:
            logger.warning("query_classifier.unexpected_labels", labels=[x["label"] for x in raw_outputs])
            return "specialized", 0.5

        if general_score >= GENERAL_CONFIDENCE_THRESHOLD:
            label, confidence = "general", general_score
        else:
            label, confidence = "specialized", 1.0 - general_score
        logger.info("query_classifier.result", text_preview=text[:50], label=label,
                    confidence=round(confidence, 4))
        return label, confidence

    # ── Data helpers (also used by scripts/evaluation/eval_classifier.py) ──

    @staticmethod
    def load_jsonl(path: str) -> list[dict]:
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                text, label = (obj.get("text") or "").strip(), obj.get("label")
                if not text:
                    raise ValueError(f"Line {line_no}: empty text")
                if label not in LABEL2ID:
                    raise ValueError(f"Line {line_no}: invalid label {label!r}")
                rows.append({"text": text, "label": label})
        if not rows:
            raise ValueError(f"No training data in {path}")
        return rows

    @staticmethod
    def stratified_split(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
        """Split each label separately so train and val keep the same label mix; fixed seed, same split."""
        rng = random.Random(seed)
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(row["label"], []).append(row)

        train_rows, val_rows = [], []
        for label in sorted(buckets):
            group = buckets[label][:]
            rng.shuffle(group)
            n_val = max(1, int(len(group) * val_ratio))
            val_rows.extend(group[:n_val])
            train_rows.extend(group[n_val:])
        rng.shuffle(train_rows)
        return train_rows, val_rows


_classifier: Optional[QueryClassifier] = None


def get_query_classifier() -> QueryClassifier:
    """Module-level singleton used by the QA nodes."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier.get_instance()
    return _classifier


TRAINING_DATA = "backend/core/data/training_data.jsonl"
HELD_OUT_DATA = "scripts/evaluation/datasets/classifier/held_out.jsonl"   # Never trained on

if __name__ == "__main__":
    # Train: python -m backend.core.query_classifier
    settings = get_settings()
    metrics = QueryClassifier(model_path=resolve_path(settings.classifier_model_path)).train(
        data_path=resolve_path(TRAINING_DATA),
        test_path=resolve_path(HELD_OUT_DATA),
        output_dir=resolve_path(settings.finetuned_classifier_path),
    )
    print({k: round(v, 4) for k, v in metrics.items() if k in ("eval_accuracy", "eval_f1_macro")})
    print(f"Saved to {settings.finetuned_classifier_path}; evaluate with python -m scripts.evaluation.eval_classifier")
