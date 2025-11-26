import ast
import json

import pandas as pd

from src.pipeline.json_extraction import load_schemes_and_labelspace
from src.pipeline.prompt_builder import PromptArgs, get_json_format_examples, build_prompt
from src.pipeline.prompts import PROMPT_TEMPLATES_MIMIC
from src.exp_args import ExpArgs
from src.utils import convert_code_to_short_code

LABELS_STR = {
    1: 'disease_vector',
    2: 'disease',
    3: 'disease',
    4: 'ICD_CODES',
}


def convert_text_to_thinking(text):
    """Remove content after json"""
    if text is None:
        return ''
    # Split the text at the first occurrence of '</think>'
    parts = text.split('</think>')[0].split('```')[0]
    if len(parts) > 1:
        return parts.strip('\n')  # Return the part before '</think>'
    return text  # If '</think>' is not found, return the original text


def manipulate_icd_json(data, icd_list, icd_version=10):
    """Filter or add entries to the ICD JSON based on the provided ICD list (labels)"""
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)
        except Exception as e:
            raise ValueError("Invalid input string: unable to parse as Python literal") from e

    # Normalize ICD codes from the list
    normalized_icd_list = {convert_code_to_short_code(code, icd_version) for code in icd_list}

    # Normalize and filter the JSON entries
    filtered_data = []
    seen_short_codes = set()

    for entry in data:
        code_no_dot = entry['icd_code'].replace('.', '')
        short_code = convert_code_to_short_code(code_no_dot, icd_version)
        if short_code in normalized_icd_list:
            filtered_data.append(entry)
            seen_short_codes.add(short_code)

    # Add missing codes from the list
    missing_codes = normalized_icd_list - seen_short_codes
    for code in icd_list:
        short_code = convert_code_to_short_code(code, icd_version)
        if short_code in missing_codes:
            filtered_data.append({'icd_code': code, 'reason': ''})

    return filtered_data


def manipulate_diagnose_pred(json_list, target_name):
    """Move diagnose with target name to the front of the list"""
    for i, item in enumerate(json_list):
        if item.get('name') == target_name:
            # Move the item to the front
            return [item] + json_list[:i] + json_list[i+1:]
    # If not found, return the list as is
    return json_list


def get_icd_thinking_string(icd_list: list[dict], thinking: str) -> str:
    """Create a thinking string from the icd list that is needed if the list is manipulated"""
    json_names = [icd['icd_code'] for icd in icd_list]
    thinking += 'I think the patient has the following ICD codes:\n'
    thinking += '\n'.join(f"{i+1}: {s}" for i, s in enumerate(json_names))
    thinking += '\nAfter a final consideration, I think the patient has the following ICD codes:\n'
    return thinking


def get_disease_thinking_string(disease_list: list[dict], thinking: str) -> str:
    """Create a thinking string from the disease list that is needed if the list is manipulated"""
    json_names = [disease['name'] for disease in disease_list]
    thinking = thinking.strip("\n")
    thinking += 'I think in order of likelihood this is the order of diseases:\n'
    thinking += '\n'.join(f"{i+1}: {s}" for i, s in enumerate(json_names))
    thinking += '\nAfter final reranking the diseases, I think this is the order of diseases:\n'
    return thinking


def create_merlin_instructions(patients: pd.DataFrame) -> pd.DataFrame:
    """Create instructions for LLM FT that either use mimic + merlin data or only mimic data"""
    instructions = pd.DataFrame()

    for i, row in patients.iterrows():
        # Loop over each verifier (v1 to v4)
        for v in range(1, 5):
            # Create Thinking data
            thinking = ''
            v_json = row.get(f'v{v}_json', "None")
            output = row[f'v{v}_json']
            if v_json != 'None':
                thinking = convert_text_to_thinking(row[f'v{v}_text'])

                # Manipulate Json so output 100% correct
                # Incorrect output will concat to the thinking
                if row[f'v{v}_score'] < 1.0:
                    v_json = ast.literal_eval(v_json)
                    if v in (2, 3):
                        thinking = get_disease_thinking_string(v_json, thinking)
                        output = str(manipulate_diagnose_pred(v_json, row[LABELS_STR[v]]))
                    elif v == 4:
                        thinking = get_icd_thinking_string(v_json, thinking)
                        output = str(manipulate_icd_json(v_json, row[LABELS_STR[v]]))

                thinking += '</think>\n'

            instructions = pd.concat([instructions, pd.DataFrame({
                'subject_id': row['subject_id'],
                'Verifier': v,
                'Score': row[f'v{v}_score'],
                'Input': row[f'v{v}_prompt'],
                'Thinking': thinking,
                'Output': output
            }, index=[0])], ignore_index=True, axis=0)
    return instructions


def get_symptom_mimic_output(symptom_dict: dict, symptom_vector: list) -> str:
    """Converts a symptom dict to a structured JSON format with value, evidence, and reasoning."""
    structured_data = {}
    symptom_names = symptom_dict.keys()
    assert len(symptom_names) == len(symptom_vector)
    for symptom, value in zip(symptom_names, symptom_vector):
        structured_data[symptom] = {
            "value": int(value),
        }
    return json.dumps(structured_data, indent=4)


def convert_mimic_output_to_json(output, p_args: PromptArgs, v_step: int):
    if v_step == 1:
        return str(get_symptom_mimic_output(p_args.manifestations_yaml, output))
    if v_step in (2, 3):
        return str({'name': output, 'reason': f'Based on the admission note {output} is suspected'})
    if v_step == 4:
        return str([
            {
                'icd_code': icd_code,
                'reason': f'Based on the admission note {icd_code} is suspected'
            } for icd_code in output
        ])


def create_mimic_instructions(patients: pd.DataFrame, exp_args: ExpArgs, p_args: PromptArgs):
    """Create instructions for LLM FT that either use mimic + merlin data or only mimic data"""
    # patients = patients.reset_index().rename(columns={'index': 'hadm_id'})
    load_schemes_and_labelspace(patients, p_args, exp_args)
    instructions = pd.DataFrame()
    prompt_data = get_json_format_examples(p_args)
    prompt_templates = ['EXTRACT_PROMPT', 'DIAGNOSE_PROMPT', 'RERANK_PROMPT', 'ICD_PROMPT']

    for i, patient in patients.iterrows():
        # Loop over each verifier (v1 to v4)
        for v in range(1, 5):
            # Create Instruction which is the prompt without merlin data
            p_args.prompt = PROMPT_TEMPLATES_MIMIC[prompt_templates[v - 1]]
            instruction = build_prompt(patient.to_dict(), p_args, prompt_data)
            output = convert_mimic_output_to_json(patient[LABELS_STR[v]], p_args, v)

            instructions = pd.concat([instructions, pd.DataFrame({
                'hadm_id': patient['hadm_id'],
                'subject_id': patient['subject_id'],
                'Verifier': v,
                'Input': instruction,
                'Output': output
            }, index=[0])], ignore_index=True, axis=0)

    file_name = f'instructions_{int(len(instructions)/4)}_mimic.pq'
    instructions.to_parquet(f'data/reasoning/abdominal_pain/instructions/{file_name}')
    return instructions
