import logging
from datetime import datetime

import numpy as np
import pandas as pd
import wandb

from src.ddx_data_gen.prompt_args import PromptArgs
from src.exp_args import ExpArgs

LABELS_STR = {
    1: 'disease_vector',
    2: 'disease',
    3: 'disease',
    4: 'ICD_CODES',
}


def log_budget_step_metrics(p_args: PromptArgs, patients: pd.DataFrame, v_step: int):
    if p_args.budget <= 1:
        return
    budget_step = p_args.budget - p_args.current_budget
    mean_score = round(np.mean(patients[f'v{v_step}_score']), 2)
    wandb.log({
        f'B_step': budget_step,
        f'v{v_step}_score': mean_score,
    })
    logging.info(f'V{v_step} Budget Step {budget_step} Mean Score: {mean_score}')


def log_dataframe(patients: pd.DataFrame, v_step: int):
    result_cols = [f'v{v_step}_prompt', f'v{v_step}_score', f'all_v{v_step}_scores',
                   f'v{v_step}_json', f'v{v_step}_text', f'v{v_step}_preds']
    v_columns = {
        0: ['Chief Complaint', 'admission_note', 'labs', 'Discharge Diagnosis'],
        1: [LABELS_STR[1]] + result_cols + ['extraction_acc'],
        2: [LABELS_STR[2]] + result_cols,
        3: [LABELS_STR[3]] + result_cols,
        4: [LABELS_STR[4]] + result_cols,
    }
    log_df = patients[v_columns[v_step]].map(str).reset_index().head(30) # Limit to 30 rows for logging
    wandb.log({f'patients_v{v_step}': wandb.Table(dataframe=log_df)})


def log_verifier_metrics(p_args: PromptArgs, exp_args: ExpArgs, patients: pd.DataFrame, v_step: int):
    mean_score = round(np.mean(patients[f'v{v_step}_score']), 2)
    minutes = round((datetime.now() - p_args.init_time).seconds / 60, 2)
    metrics = {
        f'V{v_step}_score': mean_score,
        'v_step': v_step,
        'score': mean_score,
        'used_budget': exp_args.generated_prompts,
        'minutes': minutes,
        'max_tokens': p_args.max_tokens,
        'temperature': p_args.temperature,
        'budget': p_args.budget,
        'choices': p_args.num_choices,
    }
    if 'extraction_acc' in patients.columns:
        metrics['extraction_acc'] = round(np.mean(patients['extraction_acc']), 2)
    wandb.log(metrics)
    logging.info(f'V{v_step} Top Scores:\n{patients[f"v{v_step}_score"]}')
    logging.info(f'V{v_step} Mean Score: {mean_score}')