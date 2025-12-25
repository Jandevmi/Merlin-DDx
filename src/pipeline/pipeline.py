import logging

import numpy as np
import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run

from src.exp_args import ExpArgs
from src.pipeline.json_extraction import extract_jsons_from_responses
from src.pipeline.prompt_builder import build_prompts
from src.pipeline.prompts import QUALITATIVE_EVAL_PROMPT
from src.pipeline.verifier import calculate_scores
from src.pipeline.verifier_args import VerifierArgs
from src.utils import load_sbert_model
from src.vllm_client import query_prompts, get_model
from src.wandb.data_loader import load_patients, upload_results, store_checkpoint
from src.wandb.logging import log_dataframe, log_budget_step_metrics, log_verifier_metrics, \
    log_budget_step_start, log_init
from src.wandb.run import init_wandb, update_wandb_name_tags


def get_selected_choice_index(scores: list[float], eval_mode: bool) -> int:
    """Generation mode takes choice with the highest score, eval mode takes first valid json"""
    if eval_mode:
        return next((i for i, score in enumerate(scores) if score > -1), 0)
    else:
        return np.argmax(scores)


def update_patients(
        patients: pd.DataFrame,
        work_df: pd.DataFrame,
        llm_responses: list[list],
        llm_jsons: list[list],
        scores: list[list[float]],
        preds: list,
        v_step: int,
        eval_mode: bool
):
    """Updates dataframe rows where the new score is higher than the previous best."""
    for df_i in range(len(work_df)):
        hadm_id = work_df.index[df_i]
        choice_i = get_selected_choice_index(scores[df_i], eval_mode)
        score = scores[df_i][choice_i]
        json = str(llm_jsons[df_i][choice_i])
        text = llm_responses[df_i][choice_i]

        if score > patients.loc[hadm_id, f'v{v_step}_score']:
            cols = [f'v{v_step}_score', f'all_v{v_step}_scores',
                    f'v{v_step}_json', f'v{v_step}_text', f'v{v_step}_preds']
            new_values = [score, str(scores[df_i]), json, text, str(preds[df_i][choice_i])]
            patients.loc[hadm_id, cols] = new_values
            work_df.loc[hadm_id, cols] = new_values


def filter_success_candidates(v_args: VerifierArgs, work_df: pd.DataFrame, eval_mode
                              ) -> pd.DataFrame:
    """Filters out candidates that have met the verifier threshold."""
    if eval_mode:
        return work_df[work_df[f'v{v_args.current_verifier}_json'].isin([None, 'None'])]

    if v_args.current_verifier == 3:
        # Step 3 must be at least score as good as step 2
        return work_df[(work_df[f'v3_score'] <= v_args.threshold)
                       | (work_df[f'v3_score'] < work_df[f'v2_score'])]
    else:
        return work_df[work_df[f'v{v_args.current_verifier}_score'] <= v_args.threshold]


def init_dataframe(patients: pd.DataFrame, prompts: list[str], v_step: int):
    """Initializes new columns for the verifier step if they do not exist."""
    if f'v{v_step}_score' not in patients.columns:
        patients[f'v{v_step}_score'] = -1.0
        patients[f'v{v_step}_prompt'] = prompts


async def prompt_and_evaluate(exp_args: ExpArgs, v_args: VerifierArgs, patients: pd.DataFrame,
                              work_df: pd.DataFrame, mapping_model) -> pd.DataFrame:
    # Loop until every sample reached the threshold or budget run out
    while len(work_df) and v_args.current_budget > 0:
        log_budget_step_start(v_args)
        prompts = build_prompts(v_args, work_df)
        init_dataframe(patients, prompts, v_args.current_verifier)
        vllm_response = await query_prompts(exp_args, v_args, prompts)
        extracted_json = extract_jsons_from_responses(v_args, vllm_response)
        scores, preds = calculate_scores(v_args, work_df, extracted_json, mapping_model)
        update_patients(patients, work_df, vllm_response, extracted_json,
                        scores, preds, v_args.current_verifier, exp_args.eval_mode)
        log_budget_step_metrics(v_args, patients)
        v_args.adapt_params_for_next_generation(len(work_df))
        work_df = filter_success_candidates(v_args, work_df, exp_args.eval_mode)
        store_checkpoint(exp_args, v_args, patients, work_df)

    log_verifier_metrics(exp_args, v_args, patients)
    log_dataframe(v_args, patients)
    return patients


async def qualitative_analysis(patients: pd.DataFrame, v_args: VerifierArgs, exp_args: ExpArgs):
    logging.info('Starting qualitative analysis of results.')
    i = v_args.verifier_pos
    v_args.prompt_templates[i] = QUALITATIVE_EVAL_PROMPT
    v_args.num_choices_ = 1
    v_args.max_tokens_[i] = 2000
    v_args.temperatures_[i] = 0.3
    try:
        analyses_df = patients.head(30).copy()
        prompts = build_prompts(v_args, analyses_df)
        analyses_df['Answer'] = await query_prompts(exp_args, v_args, prompts)
        log_columns = ['v4_prompt', 'v4_score', 'ICD_CODES', 'Answer']
        wandb.log({f'ICD_Analyses': wandb.Table(dataframe=analyses_df[log_columns])})
    except Exception as e:
        logging.error(f'Failed to run qualitative analysis: {e}')


async def init_pipline(v_args: VerifierArgs, exp_args: ExpArgs
                       ) -> tuple[Run, pd.DataFrame, pd.DataFrame]:
    """Initializes WandB run, loads patients, and sets label space."""
    wandb_run = init_wandb(exp_args.get_run_name(), exp_args.eval_mode)
    exp_args.llm_name = await get_model(exp_args.api_config)
    log_init(exp_args, v_args)
    update_wandb_name_tags(wandb_run, exp_args, v_args)
    patients, work_df = load_patients(wandb_run, exp_args, v_args)
    v_args.set_labelspace(exp_args, patients)
    return wandb_run, patients, work_df


async def start_pipeline(exp_args: ExpArgs, v_args: VerifierArgs):
    wandb_run, patients, work_df = await init_pipline(v_args, exp_args)
    mapping_model = load_sbert_model()
    while True:
        patients = await prompt_and_evaluate(
            exp_args, v_args, patients, work_df, mapping_model
        )
        work_df = patients.copy()
        if v_args.is_last_verifier:
            break
        v_args.set_next_verifier(exp_args.eval_mode)
    await qualitative_analysis(patients, v_args, exp_args)
    upload_results(wandb_run, exp_args, patients)
    wandb_run.finish()
