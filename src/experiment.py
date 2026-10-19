from pathlib import Path

from src.evaluator import ModelEvaluator
from src.report import ResultsReporter
from src.trainer import ModelTrainer


class ExperimentRunner:
    """Coordena modelos, folds, coleta de métricas e avaliação final no teste."""

    def __init__(self, settings, dataset):
        self.settings = settings
        self.dataset = dataset
        self.trainer = ModelTrainer(settings)
        self.evaluator = ModelEvaluator()
        self.reporter = ResultsReporter(
            settings.RESULTS_CSV,
            settings.SUMMARY_CSV,
            settings.TEST_RESULTS_CSV,
        )

    def run(self, folds: list[dict]) -> None:
        all_rows = []
        best_by_model = {}

        for model_name in self.settings.MODELS:
            model_tag = Path(model_name).stem
            best_score = float("-inf")
            best_weights = None

            for fold in folds:
                print(f"\n{'=' * 70}\nModelo: {model_name} | Fold: {fold['fold']}/{len(folds)}\n{'=' * 70}")
                weights = self.trainer.train(model_name, fold)
                metrics = self.evaluator.evaluate(weights, fold["yaml"], split="val")

                row = {
                    "model": model_tag,
                    "weights": str(weights),
                    "fold": fold["fold"],
                    "train_images": fold["train_count"],
                    "val_images": fold["val_count"],
                    **metrics,
                }
                all_rows.append(row)
                self.reporter.save_fold_results(all_rows)

                score = metrics.get("mAP50_95")
                if score is not None and score > best_score:
                    best_score = score
                    best_weights = weights

            best_by_model[model_tag] = best_weights

        self.reporter.save_summary(all_rows)

        # Avaliação final no conjunto de teste, se o YAML original o definir.
        test_source = self.dataset.get_test_source()
        if test_source is None:
            print("\nO data.yaml não define 'test'; avaliação final no teste foi ignorada.")
            return

        test_rows = []
        for model_tag, weights in best_by_model.items():
            if weights is None:
                print(f"Aviso: não foi possível selecionar checkpoint para {model_tag}.")
                continue
            # Para avaliação do teste, usamos o YAML original do dataset e split='test'.
            print(f"\nAvaliação final no teste: {model_tag} ({weights})")
            metrics = self.evaluator.evaluate(
                weights, self.dataset.data_yaml, split="test"
            )
            test_rows.append({
                "model": model_tag,
                "selected_weights": str(weights),
                "selection": "checkpoint com maior mAP50-95 na validação entre os folds",
                **metrics,
            })
        self.reporter.save_test_results(test_rows)
