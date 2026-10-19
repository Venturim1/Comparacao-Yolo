from pathlib import Path

from ultralytics import YOLO


class ModelEvaluator:
    """Calcula métricas de validação e teste com um checkpoint específico."""

    def evaluate(self, weights_path: Path, data_yaml: Path, split: str = "val") -> dict:
        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            plots=False,
            verbose=False,
        )
        values = getattr(metrics, "results_dict", {}) or {}

        def pick(*keys):
            for key in keys:
                if key in values:
                    return float(values[key])
            return None

        return {
            "precision": pick("metrics/precision(B)", "metrics/precision"),
            "recall": pick("metrics/recall(B)", "metrics/recall"),
            "mAP50": pick("metrics/mAP50(B)", "metrics/mAP50"),
            "mAP50_95": pick("metrics/mAP50-95(B)", "metrics/mAP50-95"),
        }
