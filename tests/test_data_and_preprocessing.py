import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor


class DataLoaderTests(unittest.TestCase):
    def test_binary_labels_are_normalized_explicitly(self):
        labels = pd.Series(["spam", "ham", "1", "0", "Legitimate"])
        self.assertEqual(
            DataLoader._normalize_labels(labels, "test").tolist(),
            [1, 0, 1, 0, 0],
        )

    def test_unknown_labels_fail_instead_of_being_guessed(self):
        with self.assertRaisesRegex(ValueError, "unsupported label values"):
            DataLoader._normalize_labels(pd.Series(["spam", "unsure"]), "test.csv")

    def test_kaggle_style_csv_is_loaded_with_source_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "email_dataset.csv"
            pd.DataFrame({
                "Category": ["ham", "spam"],
                "Message": ["See you at lunch", "Claim your prize now"],
            }).to_csv(path, index=False)
            frame = DataLoader(temp_dir).load_local_datasets()[0]

        self.assertEqual(frame["label"].tolist(), [0, 1])
        self.assertEqual(frame["source"].unique().tolist(), ["email_dataset"])

    def test_malformed_rows_within_one_percent_are_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "email.csv"
            categories = ["ham"] * 100 + ["spam"] * 100 + ['{"mode":"full"']
            messages = [f"message {index}" for index in range(200)] + ["isActive:false"]
            pd.DataFrame({
                "Category": categories,
                "Message": messages,
            }).to_csv(path, index=False)
            loader = DataLoader(temp_dir)
            frame = loader._standardize_csv(path)

        self.assertEqual(len(frame), 200)
        self.assertEqual(loader.skipped_rows["email.csv"]["unsupported_label_rows"], 1)

    def test_malformed_rows_over_one_percent_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "email.csv"
            pd.DataFrame({
                "Category": ["ham", "spam", '{"mode":"full"'],
                "Message": ["See you at lunch", "Claim your prize now", "isActive:false"],
            }).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "over 1%"):
                DataLoader(temp_dir)._standardize_csv(path)

    def test_combined_statistics_track_source_classes_and_cross_source_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DataLoader(temp_dir)
            uci = pd.DataFrame({
                "label": [0, 1],
                "message": ["Meet me at noon", "Claim your prize"],
                "source": ["UCI_SMS_Spam_Collection"] * 2,
            })
            csv_path = Path(temp_dir) / "email.csv"
            pd.DataFrame({
                "Category": ["ham", "spam"],
                "Message": ["Meet me at noon", "Send your bank details"],
            }).to_csv(csv_path, index=False)
            with patch.object(loader, "load_uci_sms_collection", return_value=uci):
                loader.combine_datasets()

            stats = loader.get_data_stats()

        self.assertEqual(
            stats["input_counts_by_source_before_deduplication"]["email"],
            {"legitimate": 1, "spam_or_scam": 1},
        )
        self.assertEqual(
            stats["duplicate_counts_by_source"]["email"]["repeated_from_prior_sources"],
            1,
        )


class TextPreprocessorTests(unittest.TestCase):
    def test_normalization_preserves_structural_signals(self):
        preprocessor = TextPreprocessor()
        cleaned = preprocessor.preprocess(
            "Pay ₹500 now: https://example.com and call +1 234-567-8901"
        )
        self.assertIn("inrtoken", cleaned)
        self.assertIn("urltoken", cleaned)
        self.assertIn("phonetoken", cleaned)

    def test_indicators_do_not_replace_model_predictions(self):
        indicators = TextPreprocessor().extract_scam_indicators(
            "Your prize is ready. Click https://example.com."
        )
        self.assertIn("prize", indicators)
        self.assertIn("suspicious_link", indicators)


if __name__ == "__main__":
    unittest.main()
