import json
import logging
import re
from pathlib import Path
from typing import Annotated

import pandas as pd
import yaml
from pydantic import BaseModel, create_model, ValidationError, conint, Field, RootModel

from src.pipeline.prompt_args import PromptArgs
from src.exp_args import ExpArgs
from src.utils import convert_code_to_short_code


class DiagnosisModel(BaseModel):
    name: str
    reason: str


class DiagnosesModel(
    RootModel[Annotated[list[DiagnosisModel], Field(min_length=10, max_length=10)]]):
    pass


class DiagnosesExtractModel(RootModel[list[DiagnosisModel]]):
    pass


class ICDModel(BaseModel):
    icd_code: str
    reason: str


class ICDsModel(RootModel[Annotated[list[ICDModel], Field(min_length=3, max_length=30)]]):
    pass


class ICDsExtractModel(RootModel[list[ICDModel]]):
    pass


def get_json_schema(pydantic_model):
    """Flattens a Pydantic model schema for guided decoding (removes $defs/$ref)."""
    schema = pydantic_model.model_json_schema()

    # --- Flatten internal $ref references ---
    if "$defs" in schema and "$ref" in schema.get("items", {}):
        ref_name = schema["items"]["$ref"].split("/")[-1]
        schema["items"] = schema["$defs"][ref_name]
        schema.pop("$defs", None)

    # Serialize/deserialize to make sure it’s a clean dict
    return json.loads(json.dumps(schema))


def build_model_from_yaml(name: str, yaml_data: dict[str, list], reasoning=True) -> type[BaseModel]:
    """Recursive function to create the model from the YAML"""
    def make_submodel(field_name: str, allowed_values: list) -> type[BaseModel]:
        min_val, max_val = min(allowed_values), max(allowed_values)
        value_type = conint(strict=True, ge=min_val, le=max_val)
        if reasoning:
            return create_model(
                field_name.replace(" ", "_").replace("/", "_"),
                value=(value_type, Field(..., description=f"Allowed values: {allowed_values}")),
                reasoning=(str, Field(...))
            )
        else:
            return create_model(
                field_name.replace(" ", "_").replace("/", "_"),
                value=(value_type, Field(..., description=f"Allowed values: {allowed_values}"))
                )

    fields_dict: dict[str, tuple] = {}

    for key, allowed_vals in yaml_data.items():
        submodel = make_submodel(key, allowed_vals)
        fields_dict[key] = (submodel, ...)

    # Create and return the final model class
    return create_model(name, **fields_dict)


def get_yaml_scheme(args: ExpArgs, symptom_name: str) -> (dict, BaseModel):
    yaml_path = Path(f"{args.data_dir}/medical_schemes/symptoms/{symptom_name}.yaml")

    try:
        with open(yaml_path, 'r') as f:
            yaml_schema = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f'No yaml scheme found for {symptom_name} at {yaml_path}')
        raise e

    return yaml_schema


def load_schemes_and_labelspace(patients: pd.DataFrame, prompt_args: PromptArgs, exp_args: ExpArgs):
    """Loads V1 Schema, labelspace for potential diagnoses and labelspace for ICD codes"""
    prompt_args.manifestations_yaml = get_yaml_scheme(exp_args, prompt_args.chief_complaint)
    prompt_args.pydantic_scheme = build_model_from_yaml("ManifestationsModel",
                                                        prompt_args.manifestations_yaml)
    # Mimic SFT do not have reasoning fields, thus require a separate schema
    prompt_args.manifestation_extraction_schema = build_model_from_yaml(
        "ManifestationsNoReasonModel", prompt_args.manifestations_yaml, False)

    # Wikidoc Diagnoses Labelspace
    unique_complaints = patients['Chief Complaint'].unique().tolist()
    potential_diagnoses = {}
    for chief_complaint in unique_complaints:
        print(f'Loading diagnoses for chief complaint: {chief_complaint}')
        chief_complaint = chief_complaint.replace(" ", "_")
        path = f'{exp_args.data_dir}/medical_schemes/diagnoses/{chief_complaint}.csv'
        potential_diagnoses[chief_complaint] = pd.read_csv(path)['Disease'].tolist()
    prompt_args.potential_diagnoses = potential_diagnoses

    # ICD-10 Code Labelspace
    code_lists = patients['ICD_CODES'].tolist()
    prompt_args.potential_icds = set([convert_code_to_short_code(code) for codes in code_lists
                                      for code in codes])


def truncate_reasons(diagnoses, max_length=1000):
    if not isinstance(diagnoses, list):
        return diagnoses
    for item in diagnoses:
        reason = item.get('reason')
        if isinstance(reason, str) and len(reason) > max_length:
            # Cut at the last space before max_length
            if len(reason) > max_length:
                # logging.warning(f'Truncate reason: {reason}')
                truncated = reason[:max_length].rsplit(' ', 1)[0]
                item['reason'] = truncated + '...'
    return diagnoses


def read_symptom_features(generated_output: str, schema, p_args=None) -> dict | None:
    """Extract and validate JSON-like data from LLM outputs using dynamic schemas."""
    if not generated_output:
        return None

    # --- Define regex patterns (ordered by complexity) ---
    patterns = [
        r'(\[\s*{.*?}\s*\])',             # ① JSON list of dicts
        r'(\{\s*".*?"\s*:\s*{.*}\s*\})',  # ② dict of dicts
        r'(\{[^{}]+\})',                  # ③ flat dict (ICD fallback)
    ]

    # --- Dynamic schema resolution ---
    if schema.__name__ == DiagnosesModel.__name__:
        schema = DiagnosesExtractModel
    elif schema.__name__ == ICDsModel.__name__:
        schema = ICDsExtractModel
    elif schema.__name__ == "ManifestationsModel":
        alt_schema = getattr(p_args, "manifestation_extraction_schema", None)
        schema = [schema, alt_schema] if alt_schema else [schema]

    # --- Extract all candidate JSON matches ---
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, generated_output, re.DOTALL)
        if matches and pattern == r'(\{[^{}]+\})':
            continue

        # Only wrap flat dict matches (pattern ③) for ICD-like schemas
        if pattern == r'(\{[^{}]+\})' and (
                schema is ICDsExtractModel or schema is DiagnosesExtractModel
        ):
            found = [f'[{m}]' for m in found]

        matches.extend(found)

    # Remove duplicates, keep order
    matches = list(dict.fromkeys(matches))

    # --- Try matches (reverse order: last = most complete JSON) ---
    for match in reversed(matches):
        schemas_to_try = schema if isinstance(schema, list) else [schema]
        for s in filter(None, schemas_to_try):
            try:
                json_obj = s.model_validate_json(match).model_dump()
                return truncate_reasons(json_obj)
            except ValidationError as e:
                # print(e)
                continue

    return None


def extract_json_values(data):
    values = []

    def recurse(node):
        if isinstance(node, dict):
            if "value" in node and isinstance(node["value"], (int, float)):
                values.append(node["value"])
            else:
                for val in node.values():
                    recurse(val)
        elif isinstance(node, list):
            for item in node:
                recurse(item)

    recurse(data)
    return values


def read_json_from_llm_results(llm_results: list[list[str]], prompt_args: PromptArgs) -> list:
    extracted_jsons = [[read_symptom_features(choice, prompt_args.pydantic_scheme, prompt_args)
                        for choice in choices] for choices in llm_results]
    none_count = sum(choice is not None for choices in extracted_jsons for choice in choices)
    total_count = sum(len(choices) for choices in extracted_jsons)
    logging.info(f'Extracted {none_count} valid JSONs from {total_count} total choices.')
    return extracted_jsons


def get_symptom_vectors(symptoms: list) -> list:
    return [[extract_json_values(choice) for choice in choices] for choices in symptoms]
