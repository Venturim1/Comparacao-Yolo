from pathlib import Path
from typing import Any

import yaml
from ultralytics.data.utils import check_det_dataset


class DatasetManager:
    """Lê e valida a configuração do dataset e resolve as imagens."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, data_yaml: Path):
        self.data_yaml = Path(data_yaml).expanduser().resolve()
        if not self.data_yaml.is_file():
            raise FileNotFoundError(f"Não encontrei o data.yaml: {self.data_yaml}")

        with self.data_yaml.open("r", encoding="utf-8") as f:
            self.original_config: dict[str, Any] = yaml.safe_load(f) or {}

        # A função do Ultralytics resolve path/train/val/test como o próprio
        # framework faz durante o treinamento.
        self.resolved_config = check_det_dataset(str(self.data_yaml))

    def get_development_images(self) -> list[Path]:
        """Retorna train + val, sem incluir o conjunto test."""
        images = []
        for split in ("train", "val"):
            source = self.resolved_config.get(split)
            if source is None:
                raise ValueError(
                    f"O data.yaml precisa definir '{split}' para montar o K-Fold."
                )
            images.extend(self._collect_images(source))

        unique = sorted({p.resolve() for p in images})
        if len(unique) < 2:
            raise ValueError(
                f"Foram encontradas apenas {len(unique)} imagens em train + val."
            )
        return unique

    def get_test_source(self):
        """Retorna a origem do conjunto de teste, se houver."""
        return self.resolved_config.get("test")

    def _collect_images(self, source) -> list[Path]:
        result: list[Path] = []

        if isinstance(source, (list, tuple)):
            for item in source:
                result.extend(self._collect_images(item))
            return result

        path = Path(str(source)).expanduser()
        if path.is_file() and path.suffix.lower() == ".txt":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    value = line.strip()
                    if not value:
                        continue
                    image_path = Path(value).expanduser()
                    if not image_path.is_absolute():
                        # Em listas de imagens do Ultralytics, caminhos relativos
                        # são normalmente interpretados a partir da raiz do dataset.
                        dataset_root = Path(str(self.resolved_config.get("path", self.data_yaml.parent)))
                        image_path = dataset_root / image_path
                        if not image_path.exists():
                            image_path = path.parent / value
                    if image_path.is_file() and image_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                        result.append(image_path.resolve())
            return result

        if path.is_dir():
            for image_path in path.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    result.append(image_path.resolve())
            return result

        if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
            result.append(path.resolve())
            return result

        raise FileNotFoundError(f"Origem de imagens não encontrada ou inválida: {path}")

    def make_fold_yaml(
        self,
        output_yaml: Path,
        train_list: Path,
        val_list: Path,
    ) -> Path:
        """Cria YAML do fold mantendo classes e demais metadados do YAML original."""
        output_yaml = Path(output_yaml)
        output_yaml.parent.mkdir(parents=True, exist_ok=True)

        config = dict(self.original_config)
        # Os arquivos .txt usam caminhos absolutos, evitando ambiguidades.
        config["path"] = str(output_yaml.parent.resolve())
        config["train"] = str(train_list.resolve())
        config["val"] = str(val_list.resolve())
        # Mantém 'test' no YAML original, se existir, mas não é usado no treino.
        with output_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        return output_yaml
