import logging
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExpArgs:
    data_dir: Path = Path("data")
    llm_name: str = None
    run_name: str = None
    lora: bool = False
    lora_name: str = None
    eval_mode: bool = False  # True = verifier don't use labels, False = data creation mode
    ood_eval: bool = False
    merlin_mode: bool = True
    think_about_labs: bool = True
    load_from_checkpoint: bool = False
    hardware: str = None
    config_str: dict = None
    generated_prompts: int = 0

    def __post_init__(self):
        """Post-initialization: Configure logging and warn if test mode is active."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Mimic file path: {self.data_dir / 'mimic-iv'}")

    @property
    def short_llm_name(self) -> str:
        return os.path.basename(self.llm_name)

    def get_icd_description_path(self, icd: int) -> Path:
        """Get path for ICD code description mapping."""
        return self.data_dir / "mimic-iv" / f"icd_codes_{icd}.csv"
