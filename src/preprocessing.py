"""Text normalization and TF-IDF feature engineering."""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    def __init__(self):
        self.scam_indicators = {
            "urgency": [
                "urgent", "immediately", "asap", "hurry", "act now",
                "limited time", "expire", "deadline",
            ],
            "financial": [
                "amount", "payment", "transfer", "account", "bank",
                "card", "crypto",
            ],
            "phishing": [
                "click", "link", "verify", "confirm", "update",
                "validate", "login", "password",
            ],
            "prize": [
                "won", "congratulations", "prize", "reward", "claim",
                "lottery", "jackpot", "lucky",
            ],
            "personal_information": [
                "personal", "private", "confidential", "ssn", "otp",
                "code", "pin",
            ],
        }

    @staticmethod
    def _has_term(text, term):
        return re.search(
            r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)",
            text.lower(),
        ) is not None

    def clean_text(self, text):
        """Normalize text while retaining URL, currency, and phone signals."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"(?:https?://|www\.)\S+", " urltoken ", text)
        text = re.sub(r"\b\S+@\S+\.\S+\b", " emailtoken ", text)
        text = re.sub(r"(?<!\w)\+?\d[\d ().-]{7,}\d(?!\w)", " phonetoken ", text)
        text = text.replace("₹", " inrtoken ")
        text = text.replace("$", " usdtoken ")
        text = text.replace("€", " eurtoken ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_scam_indicators(self, text):
        """Return interpretable lexical and structural warning signals."""
        if not isinstance(text, str):
            return {}
        indicators = {
            category: [
                term for term in terms if self._has_term(text, term)
            ]
            for category, terms in self.scam_indicators.items()
        }
        indicators = {category: terms for category, terms in indicators.items() if terms}

        if re.search(r"(?:https?://|www\.|bit\.ly/|tinyurl\.com/)", text, re.I):
            indicators["suspicious_link"] = ["url_present"]
        if re.search(r"(?<!\w)\+?\d[\d ().-]{7,}\d(?!\w)", text):
            indicators["phone_number"] = ["phone_present"]
        if re.search(r"[₹$€]", text):
            indicators["currency"] = ["currency_symbol_present"]
        return indicators

    def preprocess(self, text):
        return self.clean_text(text)


class FeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            max_df=0.98,
            min_df=2,
            sublinear_tf=True,
        )

    def fit_tfidf(self, texts):
        self.tfidf.fit(texts)
        return self

    def transform_tfidf(self, texts):
        return self.tfidf.transform(texts)

    def get_tfidf_feature_names(self):
        return self.tfidf.get_feature_names_out()


class DataPreprocessor:
    def __init__(self, test_size=0.2, val_size=0.1, random_state=42):
        if test_size <= 0 or val_size < 0 or test_size + val_size >= 1:
            raise ValueError("test_size must be positive and test_size + val_size < 1.")
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.feature_engineer = FeatureEngineer()

    def preprocess_data(self, df):
        required = {"message", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        df = df.copy()
        df["message"] = df["message"].fillna("").astype(str)
        df["cleaned_message"] = df["message"].apply(self.preprocessor.preprocess)
        df = df.loc[df["cleaned_message"].ne("")].copy()
        df = df.drop_duplicates(subset=["cleaned_message"], keep="first")
        if df["label"].nunique() != 2:
            raise ValueError("Binary scam detection requires both labeled classes.")
        class_counts = df["label"].value_counts()
        if class_counts.min() < 3:
            raise ValueError("Each class needs at least three unique messages for train/validation/test splits.")
        print(f"Preprocessing complete: {len(df):,} unique messages")
        return df

    def split_data(self, df):
        X = df["cleaned_message"]
        y = df["label"].astype(int)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=y,
        )
        if self.val_size == 0:
            X_val, y_val = X_temp.iloc[:0], y_temp.iloc[:0]
            X_test, y_test = X_temp, y_temp
        else:
            test_ratio = self.test_size / (self.test_size + self.val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=test_ratio,
                random_state=self.random_state,
                stratify=y_temp,
            )
        print(f"Split sizes — train: {len(X_train):,}, validation: {len(X_val):,}, test: {len(X_test):,}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def extract_features(self, X_train, X_val, X_test):
        self.feature_engineer.fit_tfidf(X_train)
        return (
            self.feature_engineer.transform_tfidf(X_train),
            self.feature_engineer.transform_tfidf(X_val),
            self.feature_engineer.transform_tfidf(X_test),
        )

    def prepare_data(self, df):
        df = self.preprocess_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        X_train_tfidf, X_val_tfidf, X_test_tfidf = self.extract_features(
            X_train, X_val, X_test
        )
        return {
            "X_train": X_train_tfidf,
            "X_val": X_val_tfidf,
            "X_test": X_test_tfidf,
            "y_train": y_train.to_numpy(),
            "y_val": y_val.to_numpy(),
            "y_test": y_test.to_numpy(),
            "feature_engineer": self.feature_engineer,
            "raw_texts_train": X_train.to_numpy(),
            "raw_texts_val": X_val.to_numpy(),
            "raw_texts_test": X_test.to_numpy(),
        }
