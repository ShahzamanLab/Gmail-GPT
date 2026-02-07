import os
from pathlib import Path

LIST_OF_FILES = [
    "src/__init__.py",
    "src/Gmail_data_loader.py",
    "src/Gmail_data_splitter.py",
    "src/Gmail_data_embeddings.py",
    "src/Gmail_data_utils.py",
    "src/Gmail_vectorstore.py",
    "src/Gmail_data_retriver.py",
    "src/Gmail_voice_rag_prompt.txt",
    "src/Gmail_text_rag.py",
    "requirements.txt",
    "token.json",
    "APP.py",
    "src/Gmail_prompt_loader.py",
    ".env"
]

for file in LIST_OF_FILES:
    path = Path(file)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.touch()
