import logging
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExpArgs:
    data_dir: Path = Path("data")
    api_config: dict = None
    lora: bool = False
    lora_name: str = None
    eval_mode: bool = False  # True = verifier don't use labels, False = data creation mode
    ood_eval: bool = False
    merlin_mode: bool = True
    think_about_labs: bool = True
    guided_decoding: bool = True
    load_from_checkpoint: bool = False
    store_patients: bool = False
    config_str: dict = None
    hardware: str = None
    llm_name: str = None
    run_name: str = None
    num_samples: int = 10
    init_time: datetime = datetime.now()

    def __post_init__(self):
        """Post-initialization: Configure logging and warn if test mode is active."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Mimic file path: {self.data_dir / 'mimic-iv'}")

    @property
    def short_llm_name(self) -> str:
        return os.path.basename(self.llm_name)

    def get_run_name(self, llm_name=None) -> str:
        if llm_name:
            return f'{self.num_samples}_{llm_name}_{str(self.init_time)[:-7]}'
        else:
            return f'{self.num_samples}_{str(self.init_time)[:-7]}'

    def get_icd_description_path(self, icd: int) -> Path:
        """Get path for ICD code description mapping."""
        return self.data_dir / "mimic-iv" / f"icd_codes_{icd}.csv"

    def get_mimic_note_path(self, note_type: str, split: str, icd: int, data_source: str) -> Path:
        """Return full path for a MIMIC-IV note file."""
        assert note_type in ["discharge", "admission"], "Invalid note type"
        assert split in ["train", "dev", "test"], "Invalid data split"
        assert data_source in ["hosp", "icu"], "Invalid data_source type"
        assert icd in [9, 10], "ICD must be 9 or 10"
        return (
                self.data_dir
                / "mimic-iv"
                / f"{note_type}_notes"
                / f"icd-{icd}"
                / data_source
                / f"{split}_{icd}_{data_source}.pq"
        )
