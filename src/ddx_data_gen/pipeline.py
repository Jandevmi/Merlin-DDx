import logging

import numpy as np
import pandas as pd
import wandb

from src.ddx_data_gen.json_extraction import read_json_from_llm_results, \
    load_schemes_and_labelspace
from src.ddx_data_gen.prompt_args import PromptArgs
from src.ddx_data_gen.prompt_builder import build_prompts
from src.ddx_data_gen.prompts import QUALITATIVE_EVAL_PROMPT
from src.ddx_data_gen.verifier import calculate_scores
from src.exp_args import ExpArgs
from src.utils import load_sbert_model
from src.vllm_client import query_prompts, get_model
from src.wandb.data_loader import load_patients, upload_results, load_checkpoint, store_checkpoint
from src.wandb.logging import log_dataframe, log_budget_step_metrics, LABELS_STR, \
    log_verifier_metrics
from src.wandb.run import init_wandb, update_wandb_name_tags


def get_json_str(jsons: list[dict]) -> str:
    return '\n'.join([f'Choice {i}: {jsons[i]}' for i in range(len(jsons)) if jsons[i] is not None])


def update_patients(
        exp_args: ExpArgs,
        patients: pd.DataFrame,
        work_df: pd.DataFrame,
        llm_responses: list[list],
        llm_jsons: list[list],
        scores: list[list[dict]],
        preds: list,
        verifier_step: int,
):
    v_columns = [f'v{verifier_step}_score', f'all_v{verifier_step}_scores',
                 f'v{verifier_step}_json', f'v{verifier_step}_text', f'v{verifier_step}_preds']

    for i in range(len(work_df)):
        hadm_id = work_df.index[i]
        v_scores = [choice['verifier_score'] for choice in scores[i]]
        if exp_args.eval_mode:
            # Take the first valid json for evaluation
            score_pos = next((i for i, score in enumerate(v_scores) if score > -1), 0)
        else:
            score_pos = np.argmax(v_scores)
        v_score = v_scores[score_pos]
        json = str(llm_jsons[i][score_pos])
        text = llm_responses[i][score_pos]
        # Log patient result to console
        if len(patients) < 20:
            label = work_df.loc[hadm_id, LABELS_STR[verifier_step]]
            json_str = get_json_str(preds[i])
            logging.info(f'{hadm_id} - Scores: {v_scores}\nLabel: {label}, JSON:\n{json_str}')

        if v_score > patients.loc[hadm_id, f'v{verifier_step}_score']:

            patients.loc[hadm_id, v_columns] = [v_score, str(v_scores), json, text, str(preds[i][score_pos])]
            work_df.loc[hadm_id, v_columns] = [v_score, str(v_scores), json, text, str(preds[i][score_pos])]

            if 'extraction_acc' in scores[i][0]:
                patients.loc[hadm_id, f'extraction_acc'] = scores[i][score_pos]['extraction_acc']


def filter_success_candidates(p_args: PromptArgs, work_df: pd.DataFrame, eval_mode, v_step) -> pd.DataFrame:
    if eval_mode:
        return work_df[work_df[f'v{v_step}_json'].isin([None, 'None'])]

    if v_step == 3:
        # Step 3 must be at least score as good as step 2
        return work_df[(work_df[f'v3_score'] <= p_args.verifier_thresholds[3])
                       | (work_df[f'v3_score'] < work_df[f'v2_score'])]
    else:
        return work_df[work_df[f'v{v_step}_score'] <= p_args.verifier_thresholds[v_step]]


def init_dataframe(patients: pd.DataFrame, prompts: list[str], v_step):
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
    logging.info(f'Starting verifier step {v_step} with {len(work_df)} candidates and budget {p_args.current_budget}')
    while len(work_df) and p_args.current_budget > 0:
        # if (p_args.current_budget >= 5) and (len(work_df) == len(patients)):
        #     break
        exp_args.generated_prompts += len(work_df)
        update_parameters(p_args, v_step, len(work_df))
        prompts = build_prompts(p_args, work_df)
        init_dataframe(patients, prompts, v_step)
        llm_response = await query_prompts(exp_args, p_args, prompts)
        extracted_json = read_json_from_llm_results(llm_response, p_args)
        scores, preds = calculate_scores(p_args, work_df, extracted_json, mapping_model)
        update_patients(exp_args, patients, work_df, llm_response, extracted_json,
                        scores, preds, v_step)
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


async def verifier_pipeline(prompt_args: PromptArgs, exp_args: ExpArgs, v_args: dict):
    wandb_run = init_wandb(prompt_args.get_run_name(), exp_args.eval_mode)
    logging.info(f'Starting verifier pipeline with v_args"\n{v_args}')
    exp_args.llm_name = await get_model(prompt_args.api_config)
    update_wandb_name_tags(wandb_run, exp_args, prompt_args)

    if exp_args.load_from_checkpoint:
        patients, work_df, v_step, budget = load_checkpoint(exp_args)
        v_step_index = v_args['verifier_steps'].index(v_step)
        v_args['verifier_steps'] = v_args['verifier_steps'][v_step_index:]
        v_args['current_budget'] = budget
    else:
        patients = load_patients(wandb_run, exp_args, prompt_args, v_args.get('start_verifier', 1))
        labs_na_string = 'No laboratory results available.'
        patients['labs'] = patients.get('labs', pd.Series()).fillna(labs_na_string)

        work_df = patients.copy()

    load_schemes_and_labelspace(patients, prompt_args, exp_args)
    mapping_model = load_sbert_model()

    for v_step in v_args['verifier_steps']:
        if exp_args.eval_mode and v_step == 3:
            logging.info('Skipping verifier step 3 in eval mode.')
            continue
        scheme = v_args['schema'][v_step - 1] or prompt_args.pydantic_scheme
        prompt_args.set_verifier_args(exp_args, v_args, v_step, scheme)
        patients = await prompt_and_evaluate(prompt_args, exp_args, patients, work_df, v_step, mapping_model)
        work_df = patients.copy()

    await qualitative_analysis(patients, prompt_args, exp_args)
    upload_results(wandb_run, exp_args, patients, v_args)

    wandb_run.finish()
