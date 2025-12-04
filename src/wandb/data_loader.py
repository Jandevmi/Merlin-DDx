import ast
import logging

import pandas as pd
import wandb

from src.pipeline.instruction_builder import create_merlin_instructions
from src.exp_args import ExpArgs
from src.pipeline.verifier_args import VerifierArgs


def load_schemes_from_wandb(wandb_run, exp_args: ExpArgs) -> dict:
    # ToDo: Implement
    file_name = f'dataset_{exp_args.num_samples}_{exp_args.short_llm_name.replace("-", "_")}'
    artifact = wandb_run.use_artifact(f"{file_name}:latest", type="dataset")


def load_cross_validation_patients(wandb_run, experiment_name: str | list) -> list[pd.DataFrame]:
    """ Load up to 3 results for an experiment from WandB artifacts for evaluation. """
    dfs = []

    if isinstance(experiment_name, str):
        experiment_name = [experiment_name]
    print(f'Download {experiment_name}')
    for i in reversed(range(20)):  # try v19 → v0
        if len(dfs) == 3:
            break
        for exp_name in experiment_name:
            if len(dfs) == 3:
                break
            artifact_ref = f"{exp_name}:v{i}"
            try:
                artifact = wandb_run.use_artifact(artifact_ref, type="eval_results")
            except (ValueError, wandb.CommError):
                logging.debug(f"{artifact_ref} not found, skipping")
                continue

            artifact_dir = artifact.download()
            file_path = f"{artifact_dir}/{exp_name}.pq"
            df = pd.read_parquet(file_path, engine="fastparquet")
            preds_cols = ['v2_preds', 'v4_preds', 'v2_json', 'v4_json']
            df[preds_cols] = df[preds_cols].map(ast.literal_eval)

            if "hadm_id" in df.columns:
                df = df.set_index("hadm_id")

            dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No artifacts found for {experiment_name}")

    return list(reversed(dfs))


def load_patients_from_wandb(wandb_run, exp_args: ExpArgs, v_step=1) -> pd.DataFrame:
    """ Load patients from WandB artifact. Files are patients_10 / patients_63 / patients_1411
    To start with v_step > 1 a run must have been completed with the llm. """
    # Fixme: Drop hadm id 22980311 for mistral model?!
    file_name = f'patients_{exp_args.num_samples}'

    if exp_args.eval_mode:
        logging.info('Loading evaluation dataset from WandB.')
        file_name = 'test_dataset'

    if exp_args.ood_eval:
        logging.info('Loading out-of-distribution evaluation dataset from WandB.')
        file_name = 'patients_ood_700'

    if v_step > 1 and not exp_args.ood_eval:
        llm_name = exp_args.short_llm_name.replace("-", "_")
        file_name = f'dataset_{exp_args.num_samples}_{llm_name}'
        artifact = wandb_run.use_artifact(f"{file_name}:latest", type="dataset")
    else:
        artifact = wandb_run.use_artifact(f"{file_name}:latest", type="dataset")

    artifact_dir = artifact.download()
    file_path = f"{artifact_dir}/{file_name}.pq"
    logging.info(f'Dataset locally stored at: {file_path}')

    return pd.read_parquet(file_path, engine='fastparquet').head(exp_args.num_samples)


def store_checkpoint(exp_args: ExpArgs, v_args: VerifierArgs, patients: pd.DataFrame,
                     work_df: pd.DataFrame):

    if exp_args.eval_mode:
        return

    path = f'/checkpoints/{exp_args.run_name}' or f'/checkpoints/default_run'
    work_path = f'{path}_work_df.pq'
    patients_path = f'{path}_patients_df.pq'

    work_df.to_parquet(work_path)
    patients.to_parquet(patients_path)
    v_args.store_checkpoint(path)


def load_checkpoint(exp_args: ExpArgs, v_args: VerifierArgs):
    path = f'/checkpoints/{exp_args.run_name}' or f'/checkpoints/default_run'
    work_path = f'{path}_work_df.pq'
    patients_path = f'{path}_patients_df.pq'
    work_df = pd.read_parquet(work_path)
    patients = pd.read_parquet(patients_path)
    v_args.load_from_checkpoint(path)

    # If checkpoint was stored at the end of a verifier step, move to next step
    if len(work_df) == 0 or v_args.current_budget == 0:
        v_args.set_next_verifier(exp_args.eval_mode)
        work_df = patients.copy()

    return patients, work_df


def load_patients(wandb_run, exp_args: ExpArgs, v_args: VerifierArgs):
    """ Try to load patients from WandB, otherwise load from local file. """
    if exp_args.load_from_checkpoint:
        return load_checkpoint(exp_args, v_args)
    else:
        try:
            patient_df = load_patients_from_wandb(wandb_run, exp_args, v_args.current_verifier)
        except Exception as e:
            if v_args.start_verifier == 1:
                logging.warning(f'Could not load patients from WandB, loading from local file: {e}')
                path = f'data/reasoning/abdominal_pain/patients_{exp_args.num_samples}.pq'
                patient_df = pd.read_parquet(path, engine='fastparquet')
            else:
                raise FileNotFoundError("Could not load patients from WandB for verifier step > 1.")

    labs_na_string = 'No laboratory results available.'
    patient_df['labs'] = patient_df.get('labs', pd.Series()).fillna(labs_na_string)

    return patient_df, patient_df.copy()


def upload_results(wandb_run, exp_args: ExpArgs, patients: pd.DataFrame):
    """ Upload patients and instructions to WandB as artifacts. """
    if exp_args.store_patients:
        instructions = None if exp_args.eval_mode else create_merlin_instructions(patients)
        upload_results_to_wandb(wandb_run, exp_args, patients, instructions)
    else:
        logging.info('Not storing patients in WandB, set store_patients to True to enable this.')


def upload_results_to_wandb(
        wandb_run,
        exp_args: ExpArgs,
        patients: pd.DataFrame,
        instructions: pd.DataFrame = None):

    if exp_args.eval_mode:
        dfs = {'eval_results': patients}
        base_name = exp_args.run_name
    else:
        dfs = {'dataset': patients, 'instructions': instructions}
        base_name = f'{len(patients)}_{exp_args.short_llm_name.replace("-", "_")}'

    for data_type, df in dfs.items():
        artifact_name = f'{data_type}_{base_name}'
        df.to_parquet(f'{artifact_name}.pq')
        artifact = wandb.Artifact(artifact_name, data_type)
        artifact.add_file(f'{artifact_name}.pq')
        wandb_run.log_artifact(artifact)
        logging.info(f'Uploaded {data_type} to WandB: {data_type}/{artifact_name}')
