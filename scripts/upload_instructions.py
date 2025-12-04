import wandb

from src.pipeline.instruction_builder import create_mimic_instructions
from src.exp_args import ExpArgs
from src.pipeline.verifier_args import VerifierArgs
from src.wandb.run import init_wandb
from src.wandb.data_loader import load_patients_from_wandb

exp_args = ExpArgs()
v_args = VerifierArgs(exp_args)
wandb_run = init_wandb("Create Mimic instructions")


def process_instructions(sample_sizes, v_step=1):
    for n in sample_sizes:
        v_args.num_samples = n
        patients = load_patients_from_wandb(wandb_run, exp_args, v_args, v_step)
        create_mimic_instructions(patients, exp_args, v_args)

        tag = f"instructions_{n}_mimic"
        file_path = f"data/reasoning/{v_args.chief_complaint}/instructions/{tag}.pq"
        artifact = wandb.Artifact(tag, type='instructions')
        artifact.add_file(file_path)
        wandb_run.log_artifact(artifact)
        print(f"Processed {n} samples → {tag}")


if __name__ == "__main__":
    process_instructions([3055])

    wandb_run.finish()
