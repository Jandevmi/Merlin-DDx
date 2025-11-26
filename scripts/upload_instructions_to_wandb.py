import wandb

from src.pipeline.instruction_builder import create_mimic_instructions
from src.pipeline.prompt_args import PromptArgs
from src.exp_args import ExpArgs
from src.wandb.run import init_wandb
from src.wandb.data_loader import load_patients_from_wandb

exp_args = ExpArgs()
prompt_args = PromptArgs(chief_complaint='abdominal_pain')
wandb_run = init_wandb("Create Mimic instructions")


def process_instructions(sample_sizes, v_step=1):
    for n in sample_sizes:
        prompt_args.num_samples = n
        patients = load_patients_from_wandb(wandb_run, exp_args, prompt_args, v_step)
        create_mimic_instructions(patients, exp_args, prompt_args)

        tag = f"instructions_{n}_mimic"
        file_path = f"data/reasoning/{prompt_args.chief_complaint}/instructions/{tag}.pq"
        artifact = wandb.Artifact(tag, type='instructions')
        artifact.add_file(file_path)
        wandb_run.log_artifact(artifact)
        print(f"Processed {n} samples → {tag}")


if __name__ == "__main__":
    process_instructions([3055])

    wandb_run.finish()
