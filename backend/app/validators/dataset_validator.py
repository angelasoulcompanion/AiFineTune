"""
Dataset Validator - Validate JSONL, CSV, Parquet files for fine-tuning
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetValidationError:
    """Represents a validation error"""

    def __init__(self, row: int, column: str, message: str, severity: str = "error"):
        self.row = row
        self.column = column
        self.message = message
        self.severity = severity

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "column": self.column,
            "message": self.message,
            "severity": self.severity,
        }


class DatasetValidationResult:
    """Result of dataset validation"""

    def __init__(self):
        self.is_valid = True
        self.errors: list[DatasetValidationError] = []
        self.warnings: list[DatasetValidationError] = []
        self.total_examples = 0
        self.train_examples = 0
        self.validation_examples = 0
        self.avg_input_length = 0
        self.avg_output_length = 0
        self.columns: list[str] = []
        self.sample_rows: list[dict] = []

    def add_error(self, row: int, column: str, message: str):
        self.errors.append(DatasetValidationError(row, column, message, "error"))
        self.is_valid = False

    def add_warning(self, row: int, column: str, message: str):
        self.warnings.append(DatasetValidationError(row, column, message, "warning"))

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "total_examples": self.total_examples,
            "train_examples": self.train_examples,
            "validation_examples": self.validation_examples,
            "avg_input_length": self.avg_input_length,
            "avg_output_length": self.avg_output_length,
            "columns": self.columns,
        }


class DatasetValidator:
    """Validates datasets for fine-tuning"""

    # Expected columns for different dataset types
    SFT_COLUMNS = {
        "required": [],  # At least one of the sets below
        "chat_format": ["messages"],  # ChatML format
        "instruction_format": ["instruction", "output"],  # Alpaca format
        "text_format": ["text"],  # Plain text format
    }

    DPO_COLUMNS = {
        "required": ["prompt", "chosen", "rejected"],
    }

    ORPO_COLUMNS = {
        "required": ["prompt", "chosen", "rejected"],
    }

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type

    def validate_file(self, file_path: str) -> DatasetValidationResult:
        """Validate a dataset file"""
        result = DatasetValidationResult()
        path = Path(file_path)

        if not path.exists():
            result.add_error(0, "", f"File not found: {file_path}")
            return result

        suffix = path.suffix.lower()

        try:
            if suffix == ".jsonl":
                self._validate_jsonl(path, result)
            elif suffix == ".json":
                self._validate_json(path, result)
            elif suffix == ".csv":
                self._validate_csv(path, result)
            elif suffix == ".parquet":
                self._validate_parquet(path, result)
            else:
                result.add_error(0, "", f"Unsupported file format: {suffix}")
        except Exception as e:
            result.add_error(0, "", f"Failed to read file: {str(e)}")
            logger.exception(f"Error validating dataset: {file_path}")

        return result

    def _validate_jsonl(self, path: Path, result: DatasetValidationResult):
        """Validate JSONL file"""
        rows = []
        input_lengths = []
        output_lengths = []

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                    rows.append(row)

                    # Validate row structure
                    self._validate_row(row, i + 1, result)

                    # Calculate lengths
                    input_len, output_len = self._calculate_lengths(row)
                    if input_len:
                        input_lengths.append(input_len)
                    if output_len:
                        output_lengths.append(output_len)

                except json.JSONDecodeError as e:
                    result.add_error(i + 1, "", f"Invalid JSON: {str(e)}")

        # Set statistics
        result.total_examples = len(rows)
        result.train_examples = len(rows)  # Will be split later
        result.validation_examples = 0
        result.avg_input_length = int(sum(input_lengths) / len(input_lengths)) if input_lengths else 0
        result.avg_output_length = int(sum(output_lengths) / len(output_lengths)) if output_lengths else 0
        result.columns = list(rows[0].keys()) if rows else []
        result.sample_rows = rows[:5]

    def _validate_json(self, path: Path, result: DatasetValidationResult):
        """Validate JSON file (array of objects)"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            result.add_error(0, "", "JSON file must contain an array of objects")
            return

        input_lengths = []
        output_lengths = []

        for i, row in enumerate(data):
            if not isinstance(row, dict):
                result.add_error(i + 1, "", "Each item must be an object")
                continue

            self._validate_row(row, i + 1, result)

            input_len, output_len = self._calculate_lengths(row)
            if input_len:
                input_lengths.append(input_len)
            if output_len:
                output_lengths.append(output_len)

        result.total_examples = len(data)
        result.train_examples = len(data)
        result.avg_input_length = int(sum(input_lengths) / len(input_lengths)) if input_lengths else 0
        result.avg_output_length = int(sum(output_lengths) / len(output_lengths)) if output_lengths else 0
        result.columns = list(data[0].keys()) if data else []
        result.sample_rows = data[:5]

    def _validate_csv(self, path: Path, result: DatasetValidationResult):
        """Validate CSV file"""
        df = pd.read_csv(path)
        self._validate_dataframe(df, result)

    def _validate_parquet(self, path: Path, result: DatasetValidationResult):
        """Validate Parquet file"""
        df = pd.read_parquet(path)
        self._validate_dataframe(df, result)

    def _validate_dataframe(self, df: pd.DataFrame, result: DatasetValidationResult):
        """Validate pandas DataFrame"""
        result.columns = df.columns.tolist()
        result.total_examples = len(df)
        result.train_examples = len(df)

        # Validate columns based on dataset type
        self._validate_columns(result.columns, result)

        # Check for empty values
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                result.add_warning(0, col, f"Column has {null_count} null values")

        # Calculate average lengths
        input_col = self._get_input_column(result.columns)
        output_col = self._get_output_column(result.columns)

        if input_col and input_col in df.columns:
            lengths = df[input_col].astype(str).str.len()
            result.avg_input_length = int(lengths.mean())

        if output_col and output_col in df.columns:
            lengths = df[output_col].astype(str).str.len()
            result.avg_output_length = int(lengths.mean())

        # Sample rows
        result.sample_rows = df.head(5).to_dict("records")

        # Validate each row
        for i, row in df.iterrows():
            self._validate_row(row.to_dict(), i + 1, result)

    def _validate_row(self, row: dict, row_num: int, result: DatasetValidationResult):
        """Validate a single row"""
        if self.dataset_type in ["sft", "chat"]:
            self._validate_sft_row(row, row_num, result)
        elif self.dataset_type == "dpo":
            self._validate_dpo_row(row, row_num, result)
        elif self.dataset_type == "orpo":
            self._validate_orpo_row(row, row_num, result)

    def _validate_sft_row(self, row: dict, row_num: int, result: DatasetValidationResult):
        """Validate SFT row"""
        # Check for valid format
        has_messages = "messages" in row
        has_instruction = "instruction" in row and "output" in row
        has_text = "text" in row

        if not (has_messages or has_instruction or has_text):
            if row_num <= 5:  # Only report first 5 errors
                result.add_error(
                    row_num, "",
                    "Row must have 'messages' (ChatML), 'instruction'+'output' (Alpaca), or 'text'"
                )
            return

        # Validate ChatML format
        if has_messages:
            messages = row["messages"]
            if not isinstance(messages, list):
                result.add_error(row_num, "messages", "messages must be an array")
            else:
                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        result.add_error(row_num, "messages", f"Message {j} is not an object")
                    elif "role" not in msg or "content" not in msg:
                        result.add_error(row_num, "messages", f"Message {j} missing 'role' or 'content'")
                    elif msg["role"] not in ["system", "user", "assistant"]:
                        result.add_warning(row_num, "messages", f"Message {j} has unknown role: {msg['role']}")

    def _validate_dpo_row(self, row: dict, row_num: int, result: DatasetValidationResult):
        """Validate DPO row"""
        for col in ["prompt", "chosen", "rejected"]:
            if col not in row:
                result.add_error(row_num, col, f"Missing required column: {col}")
            elif not row[col] or (isinstance(row[col], str) and not row[col].strip()):
                result.add_error(row_num, col, f"Column {col} is empty")

    def _validate_orpo_row(self, row: dict, row_num: int, result: DatasetValidationResult):
        """Validate ORPO row (same as DPO)"""
        self._validate_dpo_row(row, row_num, result)

    def _validate_columns(self, columns: list[str], result: DatasetValidationResult):
        """Validate that required columns exist"""
        if self.dataset_type in ["sft", "chat"]:
            has_valid = (
                "messages" in columns or
                ("instruction" in columns and "output" in columns) or
                "text" in columns
            )
            if not has_valid:
                result.add_error(
                    0, "",
                    "SFT dataset must have 'messages', 'instruction'+'output', or 'text' columns"
                )
        elif self.dataset_type in ["dpo", "orpo"]:
            for col in ["prompt", "chosen", "rejected"]:
                if col not in columns:
                    result.add_error(0, col, f"Missing required column: {col}")

    def _calculate_lengths(self, row: dict) -> tuple[int, int]:
        """Calculate input and output lengths for a row"""
        input_len = 0
        output_len = 0

        if "messages" in row:
            messages = row["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        content = str(msg["content"])
                        if msg.get("role") in ["user", "system"]:
                            input_len += len(content)
                        elif msg.get("role") == "assistant":
                            output_len += len(content)
        elif "instruction" in row:
            input_len = len(str(row.get("instruction", ""))) + len(str(row.get("input", "")))
            output_len = len(str(row.get("output", "")))
        elif "text" in row:
            input_len = len(str(row["text"]))
        elif "prompt" in row:
            input_len = len(str(row.get("prompt", "")))
            output_len = len(str(row.get("chosen", "")))

        return input_len, output_len

    def _get_input_column(self, columns: list[str]) -> str:
        """Get the input column name"""
        if "messages" in columns:
            return "messages"
        if "instruction" in columns:
            return "instruction"
        if "prompt" in columns:
            return "prompt"
        if "text" in columns:
            return "text"
        return ""

    def _get_output_column(self, columns: list[str]) -> str:
        """Get the output column name"""
        if "output" in columns:
            return "output"
        if "chosen" in columns:
            return "chosen"
        return ""


def validate_dataset(file_path: str, dataset_type: str) -> DatasetValidationResult:
    """Convenience function to validate a dataset"""
    validator = DatasetValidator(dataset_type)
    return validator.validate_file(file_path)
