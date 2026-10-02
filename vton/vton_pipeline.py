from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "vton" / "input"
OUTPUT_DIR = PROJECT_DIR / "vton" / "output"
MODELS_DIR = PROJECT_DIR / "vton" / "models"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("VTON folders ready!")
print("Input:", INPUT_DIR)
print("Output:", OUTPUT_DIR)
print("Models:", MODELS_DIR)