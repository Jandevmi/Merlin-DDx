import asyncio
import os

from src.pipeline.json_extraction import load_schemes_and_labelspace
from src.pipeline.pipeline import qualitative_analysis
from src.pipeline.prompt_args import PromptArgs
from src.exp_args import ExpArgs
from src.utils import init_notebook
from src.vllm_client import get_api_config
from src.wandb.data_loader import load_patients
from src.wandb.run import init_wandb

llm_name = "Qwen/Qwen3-32B"
num_samples = 1411

if __name__ == "__main__":
    init_notebook()
    exp_args = ExpArgs(llm_name=llm_name, )
    p_args = PromptArgs(
        api_config=get_api_config(local=True),
        num_samples=num_samples,
        num_choices=1,
    )
    wandb_run = init_wandb("Qualitative Analyses", eval_mode=False)
    patients = load_patients(wandb_run, exp_args, p_args, 4)
    load_schemes_and_labelspace(patients, p_args, exp_args)
    asyncio.run(qualitative_analysis(patients, p_args, exp_args))

