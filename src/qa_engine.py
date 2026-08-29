import csv
import json
import os
import re


class QAEngine:
    def __init__(self, data_dir, vlm_provider=None):
        self.data_dir = data_dir
        self.vlm_provider = vlm_provider

        self.log_file = os.path.join(
            data_dir,
            "continuous_count_log.csv"
        )

        self.summary_file = os.path.join(
            data_dir,
            "live_count_summary.json"
        )

        self.snapshot_dir = os.path.join(
            data_dir,
            "defect_snapshots"
        )

    def _load_summary(self):
        if not os.path.exists(self.summary_file):
            return {}

        with open(self.summary_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_records(self):
        if not os.path.exists(self.log_file):
            return []

        with open(
            self.log_file,
            "r",
            encoding="utf-8",
            newline=""
        ) as file:
            return list(csv.DictReader(file))

    def _extract_plank_id(self, question):
        patterns = [
            r"plank\s*#?\s*(\d+)",
            r"plank\s*(?:id)?\s*[:=]?\s*(\d+)"
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                question,
                re.IGNORECASE
            )

            if match:
                return int(match.group(1))

        return None

    def _find_snapshot(self, plank_id):
        if not os.path.exists(self.snapshot_dir):
            return None

        prefix = f"plank_{plank_id:04d}_"

        files = sorted(
            os.listdir(self.snapshot_dir),
            reverse=True
        )

        for filename in files:
            if filename.startswith(prefix):
                return os.path.join(
                    self.snapshot_dir,
                    filename
                )

        return None

    def answer(self, question):
        question_lower = question.lower()

        summary = self._load_summary()
        records = self._load_records()

        if not summary and not records:
            return "No inspection data is available yet."

        if (
            "total" in question_lower
            and "plank" in question_lower
        ) or "how many planks" in question_lower:

            total = summary.get(
                "running_total_count",
                len(records)
            )

            return f"The current total plank count is {total}."

        if (
            "defective" in question_lower
            and (
                "how many" in question_lower
                or "count" in question_lower
            )
        ):

            defective = summary.get(
                "running_defective_count",
                0
            )

            return f"The current defective plank count is {defective}."

        if (
            "clean" in question_lower
            and (
                "how many" in question_lower
                or "count" in question_lower
            )
        ):

            clean = summary.get(
                "running_clean_count",
                0
            )

            return f"The current clean plank count is {clean}."

        if "defect rate" in question_lower:

            rate = summary.get(
                "defect_rate_percentage",
                0
            )

            return f"The current defect rate is {rate}%."

        plank_id = self._extract_plank_id(question)

        if plank_id is not None:

            record = None

            for item in records:
                try:
                    current_id = int(item.get("Plank_ID", -1))
                except ValueError:
                    continue

                if current_id == plank_id:
                    record = item
                    break

            snapshot = self._find_snapshot(plank_id)

            if snapshot and self.vlm_provider:
                return self.vlm_provider.ask(
                    question,
                    snapshot
                )

            if record:
                status = record.get(
                    "Status",
                    "UNKNOWN"
                )

                defects = record.get(
                    "Defect_Types",
                    "None"
                )

                return (
                    f"Plank #{plank_id} is {status}. "
                    f"Detected defects: {defects}."
                )

            return f"No record was found for plank #{plank_id}."

        if self.vlm_provider:
            return self.vlm_provider.ask(question)

        return (
            "I can answer questions about total planks, "
            "clean planks, defective planks, defect rate, "
            "and individual plank records."
        )