from pathlib import Path

from ultralytics import YOLO


class ModelTrainer:
    """Treina uma arquitetura em um fold e devolve o caminho do melhor checkpoint."""

    def __init__(self, settings):
        self.settings = settings

    def train(self, model_name: str, fold: dict) -> Path:
        model_tag = Path(model_name).stem
        run_name = f"{model_tag}_fold{fold['fold']}"

        model = YOLO(model_name)
        model.train(
            data=str(fold["yaml"]),
            epochs=self.settings.EPOCHS,
            imgsz=self.settings.IMAGE_SIZE,
            batch=self.settings.BATCH_SIZE,
            workers=self.settings.WORKERS,
            patience=self.settings.PATIENCE,
            device=self.settings.DEVICE,
            seed=self.settings.SEED,
            cache=self.settings.CACHE,
            plots=self.settings.PLOTS,
            project=str(self.settings.TRAINING_PROJECT_DIR),
            name=run_name,
            exist_ok=True,
            verbose=self.settings.VERBOSE,
        )

        best_path = self.settings.TRAINING_PROJECT_DIR / run_name / "weights" / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Não encontrei best.pt em {best_path}. Verifique o log do treinamento."
            )
        return best_path
