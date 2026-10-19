from pathlib import Path

import pandas as pd


class ResultsReporter:
    """Salva métricas por fold, resumo estatístico e resultados de teste."""

    METRICS = ["precision", "recall", "mAP50", "mAP50_95"]

    def __init__(self, results_csv: Path, summary_csv: Path, test_csv: Path):
        self.results_csv = Path(results_csv)
        self.summary_csv = Path(summary_csv)
        self.test_csv = Path(test_csv)

    def save_fold_results(self, rows: list[dict]) -> None:
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(self.results_csv, index=False, encoding="utf-8-sig")

    def save_summary(self, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        if df.empty:
            return
        aggregations = {metric: ["mean", "std"] for metric in self.METRICS if metric in df.columns}
        summary = df.groupby("model").agg(aggregations)
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        self.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(self.summary_csv, index=False, encoding="utf-8-sig")

    def save_test_results(self, rows: list[dict]) -> None:
        self.test_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(self.test_csv, index=False, encoding="utf-8-sig")
