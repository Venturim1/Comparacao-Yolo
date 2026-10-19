from pathlib import Path

from sklearn.model_selection import KFold

from src.dataset import DatasetManager


class CrossValidationManager:
    """Gera os mesmos folds uma vez para reutilizá-los em todas as arquiteturas."""

    def __init__(self, dataset: DatasetManager, n_splits: int, random_state: int, folds_dir: Path):
        self.dataset = dataset
        self.n_splits = n_splits
        self.random_state = random_state
        self.folds_dir = Path(folds_dir)

    def create_folds(self) -> list[dict]:
        images = self.dataset.get_development_images()
        if self.n_splits < 2:
            raise ValueError("N_SPLITS precisa ser pelo menos 2.")
        if len(images) < self.n_splits:
            raise ValueError(
                f"Há {len(images)} imagens, menos que N_SPLITS={self.n_splits}."
            )

        self.folds_dir.mkdir(parents=True, exist_ok=True)
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = []

        for fold_number, (train_idx, val_idx) in enumerate(kfold.split(images), start=1):
            fold_dir = self.folds_dir / f"fold_{fold_number}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_list = fold_dir / "train.txt"
            val_list = fold_dir / "val.txt"
            self._write_paths(train_list, [images[i] for i in train_idx])
            self._write_paths(val_list, [images[i] for i in val_idx])

            fold_yaml = self.dataset.make_fold_yaml(
                fold_dir / "data.yaml", train_list, val_list
            )
            folds.append({
                "fold": fold_number,
                "train_list": train_list,
                "val_list": val_list,
                "yaml": fold_yaml,
                "train_count": len(train_idx),
                "val_count": len(val_idx),
            })

        return folds

    @staticmethod
    def _write_paths(file_path: Path, images: list[Path]) -> None:
        with file_path.open("w", encoding="utf-8") as f:
            for image_path in images:
                f.write(str(image_path.resolve()) + "\n")
