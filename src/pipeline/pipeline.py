import logging

import numpy as np
import pandas as pd
import wandb

from src.exp_args import ExpArgs
from src.pipeline.json_extraction import read_json_from_llm_results, load_schemes_and_labelspace
from src.pipeline.prompt_args import PromptArgs
from src.pipeline.prompt_builder import build_prompts
from src.pipeline.prompts import QUALITATIVE_EVAL_PROMPT
from src.pipeline.verifier import calculate_scores
from src.utils import load_sbert_model
from src.vllm_client import query_prompts, get_model
from src.wandb.data_loader import load_patients, upload_results, store_checkpoint
from src.wandb.logging import log_dataframe, log_budget_step_metrics, log_verifier_metrics
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


def filter_success_candidates(p_args: PromptArgs, work_df: pd.DataFrame, eval_mode, v_step
                              ) -> pd.DataFrame:
    """Filters out candidates that have met the verifier threshold."""
    if eval_mode:
        return work_df[work_df[f'v{v_step}_json'].isin([None, 'None'])]

    if v_step == 3:
        # Step 3 must be at least score as good as step 2
        return work_df[(work_df[f'v3_score'] <= p_args.verifier_thresholds[3])
                       | (work_df[f'v3_score'] < work_df[f'v2_score'])]
    else:
        return work_df[work_df[f'v{v_step}_score'] <= p_args.verifier_thresholds[v_step]]


def init_dataframe(patients: pd.DataFrame, prompts: list[str], v_step):
    """Initializes new columns for the verifier step if they do not exist."""
    if f'v{v_step}_score' not in patients.columns:
        patients[f'v{v_step}_score'] = -1.0
        patients[f'v{v_step}_prompt'] = prompts


def update_parameters(p_args: PromptArgs, v_step: int, df_len: int):
    """Temp is increase each round to a limit. Remains 0.0 for guided decoding"""
    if p_args.temperature <= 1.4 and not p_args.guided_decoding:
        p_args.temperature += 0.2
    p_args.max_tokens += 200
    logging.info(f'V{v_step}: Samples: {df_len}, Budget: {p_args.current_budget}, '
                 f'Temp: {p_args.temperature}, Max Tokens: {p_args.max_tokens}')


async def prompt_and_evaluate(p_args: PromptArgs, exp_args: ExpArgs, patients: pd.DataFrame, work_df: pd.DataFrame,
                              v_step: int, mapping_model) -> pd.DataFrame:
    # Loop until every sample reached the threshold
    logging.info(f'Starting verifier step {v_step} with {len(work_df)} patients and budget {p_args.current_budget}')
    while len(work_df) and p_args.current_budget > 0:
        exp_args.generated_prompts += len(work_df)
        update_parameters(p_args, v_step, len(work_df))
        prompts = build_prompts(p_args, work_df)
        init_dataframe(patients, prompts, v_step)
        llm_response = await query_prompts(exp_args, p_args, prompts)
        extracted_json = read_json_from_llm_results(llm_response, p_args)
        scores, preds = calculate_scores(p_args, work_df, extracted_json, mapping_model)
        update_patients(patients, work_df, llm_response, extracted_json,
                        scores, preds, v_step, exp_args.eval_mode)
        log_budget_step_metrics(p_args, patients, v_step)
        work_df = filter_success_candidates(p_args, work_df, exp_args.eval_mode, v_step)
        p_args.current_budget -= 1
        store_checkpoint(patients, work_df, v_step, p_args.current_budget, exp_args)

    log_verifier_metrics(p_args, exp_args, patients, v_step)
    log_dataframe(patients, v_step)
    return patients


async def qualitative_analysis(patients: pd.DataFrame, p_args: PromptArgs, exp_args: ExpArgs):
    logging.info('Starting qualitative analysis of results.')
    p_args.prompt = QUALITATIVE_EVAL_PROMPT
    p_args.num_choices = 1
    p_args.max_tokens = 2000
    p_args.temperature = 0.3
    try:
        analyses_df = patients.head(30).copy()
        prompts = build_prompts(p_args, analyses_df)
        analyses_df['Answer'] = await query_prompts(exp_args, p_args, prompts)
        log_columns = ['v4_prompt', 'v4_score', 'ICD_CODES', 'Answer']
        wandb.log({f'ICD_Analyses': wandb.Table(dataframe=analyses_df[log_columns])})
    except Exception as e:
        logging.error(f'Failed to run qualitative analysis: {e}')


async def init_pipline(prompt_args: PromptArgs, exp_args: ExpArgs, v_args: dict):
    wandb_run = init_wandb(prompt_args.get_run_name(), exp_args.eval_mode)
    exp_args.llm_name = await get_model(prompt_args.api_config)
    update_wandb_name_tags(wandb_run, exp_args, prompt_args)
    patients, work_df = load_patients(wandb_run, exp_args, prompt_args, v_args)
    load_schemes_and_labelspace(patients, prompt_args, exp_args)
    mapping_model = load_sbert_model()
    return wandb_run, patients, work_df, mapping_model


async def start_pipeline(p_args: PromptArgs, exp_args: ExpArgs, v_args: dict):
    """Initiates pipeline and iterates over each verifier step."""
    wandb_run, patients, work_df, mapping_model = await init_pipline(p_args, exp_args, v_args)

    for v_step in v_args['verifier_steps']:
        # Adapt for eval pipeline
        if exp_args.eval_mode and v_step == 3:
            logging.info('Skipping verifier step 3 in eval mode.')
            continue

        p_args.set_verifier_args(exp_args, v_args, v_step)
        patients = await prompt_and_evaluate(p_args, exp_args, patients, work_df, v_step,
                                             mapping_model)
        work_df = patients.copy()

    await qualitative_analysis(patients, p_args, exp_args)
    upload_results(wandb_run, exp_args, patients, v_args)

    wandb_run.finish()
