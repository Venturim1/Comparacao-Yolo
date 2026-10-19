from config import settings
from src.cross_validation import CrossValidationManager
from src.dataset import DatasetManager
from src.experiment import ExperimentRunner


def main():
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = DatasetManager(settings.DATA_YAML)
    folds_manager = CrossValidationManager(
        dataset=dataset,
        n_splits=settings.N_SPLITS,
        random_state=settings.RANDOM_STATE,
        folds_dir=settings.FOLDS_DIR,
    )
    folds = folds_manager.create_folds()

    print(f"Imagens de desenvolvimento (train + val): "
          f"{sum(f['train_count'] + f['val_count'] for f in folds) // settings.N_SPLITS}")
    print(f"Número de folds: {len(folds)}")
    print(f"Modelos: {', '.join(settings.MODELS)}")
    print(f"Saída: {settings.OUTPUT_DIR}")

    runner = ExperimentRunner(settings, dataset)
    runner.run(folds)


if __name__ == "__main__":
    main()
