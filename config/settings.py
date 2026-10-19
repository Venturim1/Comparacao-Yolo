from pathlib import Path

# Caminho do data.yaml original do dataset.
# Use caminho absoluto para evitar dúvidas de resolução de caminhos.
DATA_YAML = Path(r"C:\caminho\para\D-Fire\data.yaml")

# Experimento
MODELS = ["yolov8s.pt", "yolov10s.pt", "yolo26s.pt"]
N_SPLITS = 5
RANDOM_STATE = 42

# Treinamento
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 24
WORKERS = 2
DEVICE = "0"  # Use "cpu" para CPU; em Linux/Windows com CUDA, normalmente "0".
SEED = 42
CACHE = False
PLOTS = True

# Pastas e parâmetros de execução
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "runs" / "kfold_tcc"
FOLDS_DIR = OUTPUT_DIR / "folds"
TRAINING_PROJECT_DIR = OUTPUT_DIR / "training"
RESULTS_CSV = OUTPUT_DIR / "results_per_fold.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
TEST_RESULTS_CSV = OUTPUT_DIR / "test_results.csv"

# O treinamento pode ser retomado manualmente, mas o script não tenta
# retomar checkpoints automaticamente.
VERBOSE = True
