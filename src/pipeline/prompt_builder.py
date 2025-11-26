import ast
import json

import pandas as pd

from src.pipeline.json_extraction import DiagnosesModel, ICDsModel
from src.pipeline.prompt_args import PromptArgs


def get_symptom_format_example(symptom_dict) -> str:
    """Converts a symptom dict to a structured JSON format with value, evidence, and reasoning."""
    structured_data = {}
    for symptom, _ in symptom_dict.items():
        structured_data[symptom] = {
            "value": 0,
            "reasoning": f"Based on the admission note, {symptom} is suspected to be present / "
                         f"absent / unmentioned."
        }
    return json.dumps(structured_data, indent=4)


def get_diagnose_format_example() -> str:
    return '''[
        {
            "name": "diagnosis_0",
            "reason": "Based on clinical evidence, diagnosis_0 is suspected."
        },
        {
            "name": "diagnosis_1",
            "reason": "Based on clinical evidence, diagnosis_1 is suspected."
        },
        ...
        {
            "name": "diagnosis_9",
            "reason": "Based on clinical evidence, diagnosis_9 is suspected."
        }
    ]'''


def get_icd_format_example() -> str:
    return '''[
    {
        "icd_code": "icd_code_0",
        "reason": "Based on clinical evidence, icd_code_0 is suspected."
    },
    {
        "icd_code": "icd_code_1",
        "reason": "Based on clinical evidence, icd_code_1 is suspected."
    },
    ...
]'''


def get_json_format_examples(p_args: PromptArgs) -> dict:
    return {
        'symptom_format_example': get_symptom_format_example(p_args.manifestations_yaml),
        'diagnose_format_example': get_diagnose_format_example(),
        'icd_format_example': get_icd_format_example(),
    }


def format_manifestations_from_string(dict_str: str) -> str:
    try:
        data = ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return f"Invalid dictionary string provided: {dict_str}"

    present_section = ["\n#### Present Manifestations"]
    absent_section = ["\n#### Absent Manifestations"]

    for manifestation, details in data.items():
        formatted_entry = (
            f"  - {manifestation}: {details.get('reasoning', 'N/A')}"
        )
        if details.get("value") == 1:
            present_section.append(formatted_entry)
        elif details.get("value") == -1:
            absent_section.append(formatted_entry)

    return "\n".join(present_section + absent_section)


def get_diagnose_string(value, label, fallback):
    """Convert stringified JSON if valid, otherwise return fallback."""
    diagnoses = ast.literal_eval(value)
    if value in ("None", None):
        return fallback
    else:
        diagnose_string = ""
        for i, d in enumerate(diagnoses):
            diagnose_string += f"  {i+1}. {d['name']}: {d.get('reason', 'No reason provided.')}\n"
            if i > 4 and d['name'] == label:
                break
        return diagnose_string


def build_prompt(patient: dict, p_args: PromptArgs, prompt_data: dict) -> str:
    # Manifestations
    v1_json = patient.get('v1_json', "None")
    manifestation_string = (
        format_manifestations_from_string(v1_json)
        if v1_json not in ("None", None)
        else "Failed to extract manifestations"
    )

    # Diagnoses
    v2_json = patient.get('v2_json', "None")
    v3_json = patient.get('v3_json', v2_json)

    diagnose_label = patient['disease']
    potential_diagnoses = p_args.potential_diagnoses[patient['Chief Complaint']]
    diagnose_string = get_diagnose_string(v2_json, diagnose_label, potential_diagnoses)

    if p_args.pydantic_scheme == ICDsModel:
        diagnose_string = get_diagnose_string(v3_json, diagnose_label, "Not assigned")

    # Build prompt
    prompt_data.update({
        'admission_note': patient['admission_note'],
        'laboratory_results': patient['labs'],
        'potential_diagnoses': diagnose_string,
        'clinical_manifestations': manifestation_string,
        # For qualitative analysis
        'v4_prompt': patient.get('v4_prompt', "None"),
        'v4_json': patient.get('v4_json', "None"),
        'ICD_CODES': str(patient.get('ICD_CODES', "None")),
    })

    return p_args.prompt.format(**prompt_data)


def extract_admission_note_from_prompt(text: str) -> str:
    idx = text.find("### Admission Note")
    if idx != -1:
        return text[idx:]
    else:
        return text  # fallback if not found


def build_prompts(p_args: PromptArgs, patients: pd.DataFrame) -> list[str]:
    prompt_data = get_json_format_examples(p_args)
    return [build_prompt(patients_dict, p_args, prompt_data)
            for patients_dict in patients.to_dict(orient="records")]
