import os
from pathlib import Path

LIST_OF_FILES = [
    "src/Gmail_data_loader.py",
    "src/Gmail_data_preprocessing.py",
    "src/Gmail_data_brain.py",
    "requirements.txt",
    ".env"
]

for file in LIST_OF_FILES:
    path = Path(file)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.touch()
