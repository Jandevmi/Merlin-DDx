import logging
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def init_notebook(depth=0):
    """Initialize working directory and logging for Jupyter notebooks."""
    top_level = os.getenv("REPO_ROOT", str(Path(os.getcwd()).parents[depth]))
    os.environ["REPO_ROOT"] = top_level
    os.chdir(top_level)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Working directory set to: {top_level}")


def load_sbert_model(model_name='neuml/pubmedbert-base-embeddings'):
    """Load a Sentence-BERT model from HuggingFace or local directory (PVC)."""
    # print(f'Looking for model folder {[x[0] for x in os.walk('.')]}')
    if os.path.isdir(f'/models/{model_name}'):
        model_name = f'/models/{model_name}'
        logging.info(f'Loading local model from {model_name}')
    else:
        logging.info(f'Loading model {model_name} from HuggingFace')
    return SentenceTransformer(model_name, device=detect_device())


def convert_code_to_short_code(code: list, icd_version: int = 10) -> str:
    """Converts ICD codes to short format."""
    return code[:4] if icd_version == 9 and code.lower().startswith('e') else code[:3]


def convert_codes_to_short_codes(codes: list, icd_version: int = 10) -> list:
    """Batch conversion of codes to short format."""
    return list(set([convert_code_to_short_code(code, icd_version) for code in codes]))


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # logging.info(f'Found device {device}')
    return device
