"""Load verified, labeled spam-message datasets for ScamText AI."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd
import requests


UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
POSITIVE_LABELS = {
    "spam", "scam", "phishing", "fraud", "malicious", "smishing", "junk",
    "fraudulent", "financial scam", "job scam", "investment scam",
    "lottery scam", "account scam", "other scam",
    "1", "yes", "true",
}
NEGATIVE_LABELS = {
    "ham", "legitimate", "not spam", "not_spam", "non-spam", "non spam",
    "benign", "safe", "normal", "0", "no", "false",
}
LABEL_MAP = {
    **{label: 1 for label in POSITIVE_LABELS},
    **{label: 0 for label in NEGATIVE_LABELS},
}
MESSAGE_COLUMNS = ("message", "text", "body", "email", "sms", "content", "v2")
LABEL_COLUMNS = ("label", "category", "class", "target", "spam", "v1")


class DataLoader:
    """Load UCI SMS data and user-provided Kaggle CSVs with verified labels."""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.combined_data = None
        self.raw_sample_count = 0
        self.duplicate_count = 0
        self.raw_class_counts = {}
        self.raw_source_class_counts = {}
        self.skipped_rows = {}
        self.source_duplicate_counts = {}

    @staticmethod
    def _normalize_labels(labels, source):
        normalized = labels.astype(str).str.strip().str.lower()
        normalized = normalized.replace({"0.0": "0", "1.0": "1"})
        mapped = normalized.map(LABEL_MAP)
        unknown = sorted(normalized[mapped.isna()].unique().tolist())
        if unknown:
            raise ValueError(
                f"{source} contains unsupported label values: {unknown[:10]}. "
                "Use explicit spam/scam and ham/legitimate labels, or 0/1."
            )
        return mapped.astype("int64")

    @staticmethod
    def _find_column(columns, candidates):
        by_name = {str(column).strip().lower(): column for column in columns}
        return next((by_name[name] for name in candidates if name in by_name), None)

    def _standardize_csv(self, path):
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"{path.name} is empty.")

        message_column = self._find_column(frame.columns, MESSAGE_COLUMNS)
        label_column = self._find_column(frame.columns, LABEL_COLUMNS)

        if message_column is None or label_column is None:
            if len(frame.columns) == 2:
                first, second = frame.columns
                first_values = set(frame[first].dropna().astype(str).str.strip().str.lower().unique())
                second_values = set(frame[second].dropna().astype(str).str.strip().str.lower().unique())
                if first_values and first_values <= (POSITIVE_LABELS | NEGATIVE_LABELS):
                    label_column, message_column = first, second
                elif second_values and second_values <= (POSITIVE_LABELS | NEGATIVE_LABELS):
                    label_column, message_column = second, first

        if message_column is None or label_column is None:
            raise ValueError(
                f"{path.name} needs a message/text column and a label/category column. "
                f"Found columns: {list(frame.columns)}"
            )

        messages = frame[message_column].fillna("").astype(str).str.strip()
        normalized_labels = frame[label_column].astype(str).str.strip().str.lower()
        normalized_labels = normalized_labels.replace({"0.0": "0", "1.0": "1"})
        labels = normalized_labels.map(LABEL_MAP)
        invalid_labels = labels.isna()
        invalid_count = int(invalid_labels.sum())
        if invalid_count * 100 > len(frame):
            invalid_values = sorted(
                normalized_labels[invalid_labels].unique().tolist()
            )
            raise ValueError(
                f"{path.name} has {invalid_count} rows with unsupported labels "
                f"{invalid_values[:10]}; over 1% of rows is not safe to exclude."
            )

        empty_messages = messages.eq("")
        empty_message_count = int((empty_messages & ~invalid_labels).sum())
        skipped_count = invalid_count + empty_message_count
        if invalid_count:
            invalid_values = normalized_labels[invalid_labels].value_counts().to_dict()
            print(
                f"Warning: excluding {invalid_count} rows in {path.name} with "
                f"non-binary labels {invalid_values}; no labels were inferred."
            )
        if skipped_count:
            self.skipped_rows[path.name] = {
                "rows": skipped_count,
                "unsupported_label_rows": invalid_count,
                "empty_message_rows": empty_message_count,
            }
        usable = ~invalid_labels & ~empty_messages
        result = pd.DataFrame({
            "label": labels[usable].to_numpy(),
            "message": messages[usable].to_numpy(),
            "source": path.stem,
        })
        if result.empty:
            raise ValueError(f"{path.name} contains no non-empty labeled messages.")
        return result

    def load_uci_sms_collection(self):
        """Read the UCI corpus from a local archive or download its public ZIP."""
        corpus_path = self.data_dir / "SMSSpamCollection"
        zip_path = self.data_dir / "sms_spam.zip"

        if corpus_path.exists():
            frame = pd.read_csv(
                corpus_path, sep="\t", header=None, names=["label", "message"],
                encoding="latin-1",
            )
        else:
            if not zip_path.exists():
                partial_path = self.data_dir / "sms_spam.zip.part"
                try:
                    with requests.get(UCI_URL, stream=True, timeout=(10, 60)) as response:
                        response.raise_for_status()
                        with partial_path.open("wb") as archive_file:
                            for chunk in response.iter_content(chunk_size=1024 * 64):
                                if chunk:
                                    archive_file.write(chunk)
                    os.replace(partial_path, zip_path)
                except requests.RequestException as exc:
                    partial_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        "Could not download the UCI SMS Spam Collection. "
                        "Check the internet connection or place the official "
                        "SMSSpamCollection file or sms_spam.zip in the data folder."
                    ) from exc

            try:
                with zipfile.ZipFile(zip_path) as archive:
                    member = next(
                        (name for name in archive.namelist()
                         if Path(name).name == "SMSSpamCollection"),
                        None,
                    )
                    if member is None:
                        raise ValueError("UCI archive does not contain SMSSpamCollection.")
                    with archive.open(member) as corpus_file:
                        frame = pd.read_csv(
                            corpus_file, sep="\t", header=None,
                            names=["label", "message"], encoding="latin-1",
                        )
            except (OSError, zipfile.BadZipFile) as exc:
                raise ValueError(f"Unable to read UCI archive {zip_path}: {exc}") from exc

        frame["label"] = self._normalize_labels(frame["label"], "UCI SMS Spam Collection")
        frame["message"] = frame["message"].fillna("").astype(str).str.strip()
        frame = frame.loc[frame["message"].ne("")].copy()
        frame["source"] = "UCI_SMS_Spam_Collection"
        print(f"Loaded UCI SMS Spam Collection: {len(frame):,} labeled messages")
        return frame[["label", "message", "source"]]

    def load_local_datasets(self):
        """Load user-provided CSVs from data/ and the project root."""
        frames = []
        folders = [self.data_dir]
        if self.data_dir.name.lower() == "data":
            folders.append(self.data_dir.parent)
        paths = sorted({
            path.resolve()
            for folder in folders
            for path in folder.glob("*.csv")
        })
        for path in paths:
            if path.name.lower() == "combined_dataset.csv":
                continue
            frame = self._standardize_csv(path)
            print(f"Loaded local dataset {path.name}: {len(frame):,} labeled messages")
            frames.append(frame)
        return frames

    def combine_datasets(self):
        """Combine real data sources; never create synthetic fallback examples."""
        frames = [self.load_uci_sms_collection()]
        frames.extend(self.load_local_datasets())
        combined = pd.concat(frames, ignore_index=True)
        combined["message_key"] = combined["message"].str.strip().str.casefold()
        self.raw_sample_count = int(len(combined))
        self.raw_class_counts = {
            int(label): int(count)
            for label, count in combined["label"].value_counts().items()
        }
        self.raw_source_class_counts = {}
        for (source, label), count in combined.groupby(["source", "label"]).size().items():
            self.raw_source_class_counts.setdefault(str(source), {})[int(label)] = int(count)
        seen_messages = set()
        self.source_duplicate_counts = {}
        for source, source_rows in combined.groupby("source", sort=False):
            source_messages = set(source_rows["message_key"])
            repeated_within_source = int(len(source_rows) - len(source_messages))
            repeated_from_prior_sources = int(len(source_messages & seen_messages))
            self.source_duplicate_counts[str(source)] = {
                "repeated_within_source": repeated_within_source,
                "repeated_from_prior_sources": repeated_from_prior_sources,
                "unique_new_messages": int(len(source_messages - seen_messages)),
            }
            seen_messages.update(source_messages)
        self.duplicate_count = int(combined["message_key"].duplicated().sum())
        combined = combined.drop_duplicates("message_key", keep="first")
        self.combined_data = combined.drop(columns="message_key").reset_index(drop=True)

        print(
            f"Combined {len(self.combined_data):,} unique messages "
            f"({self.duplicate_count:,} exact duplicates removed from "
            f"{self.raw_sample_count:,} labeled rows)."
        )
        print(f"Class counts: {self.combined_data['label'].value_counts().sort_index().to_dict()}")
        print(f"Sources: {self.combined_data['source'].value_counts().to_dict()}")
        return self.combined_data

    def save_combined_data(self, filename="combined_dataset.csv"):
        if self.combined_data is None:
            raise RuntimeError("No data loaded. Call combine_datasets() first.")
        filepath = self.data_dir / filename
        self.combined_data.to_csv(filepath, index=False)
        print(f"Combined dataset saved to {filepath}")

    def get_data_stats(self):
        if self.combined_data is None:
            raise RuntimeError("No data loaded. Call combine_datasets() first.")
        lengths = self.combined_data["message"].str.len()
        stats = {
            "samples": int(len(self.combined_data)),
            "input_samples_before_deduplication": self.raw_sample_count,
            "duplicates_removed": self.duplicate_count,
            "input_class_counts_before_deduplication": {
                "legitimate": self.raw_class_counts.get(0, 0),
                "spam_or_scam": self.raw_class_counts.get(1, 0),
            },
            "input_counts_by_source_before_deduplication": {
                source: {
                    "legitimate": counts.get(0, 0),
                    "spam_or_scam": counts.get(1, 0),
                }
                for source, counts in self.raw_source_class_counts.items()
            },
            "skipped_source_rows": self.skipped_rows,
            "duplicate_counts_by_source": self.source_duplicate_counts,
            "classes": {
                "legitimate": int((self.combined_data["label"] == 0).sum()),
                "spam_or_scam": int((self.combined_data["label"] == 1).sum()),
            },
            "sources": {
                str(source): int(count)
                for source, count in self.combined_data["source"].value_counts().items()
            },
            "message_length_characters": {
                "mean": float(lengths.mean()),
                "median": float(lengths.median()),
                "minimum": int(lengths.min()),
                "maximum": int(lengths.max()),
            },
        }
        print(f"Dataset statistics: {stats}")
        return stats


if __name__ == "__main__":
    loader = DataLoader()
    loader.combine_datasets()
    loader.save_combined_data()
    loader.get_data_stats()
