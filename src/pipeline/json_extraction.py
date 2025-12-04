import logging
import re

from pydantic import ValidationError

from src.pipeline.pydantic_schemas import DiagnosesModel, DiagnosesExtractModel, ICDsModel, \
    ICDsExtractModel
from src.pipeline.verifier_args import VerifierArgs


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


def read_symptom_features(generated_output: str, schema) -> dict | None:
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

    # Try matches (reverse order: last = final JSON)
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


def read_json_from_llm_results(llm_results: list[list[str]], v_args: VerifierArgs) -> list:
    extracted_jsons = [[read_symptom_features(choice, v_args.pydantic_scheme)
                        for choice in choices] for choices in llm_results]
    none_count = sum(choice is not None for choices in extracted_jsons for choice in choices)
    total_count = sum(len(choices) for choices in extracted_jsons)
    logging.info(f'Extracted {none_count} valid JSONs from {total_count} total choices.')
    return extracted_jsons


def get_symptom_vectors(symptoms: list) -> list:
    return [[extract_json_values(choice) for choice in choices] for choices in symptoms]
