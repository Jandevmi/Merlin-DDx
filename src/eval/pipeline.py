import pandas as pd
from sentence_transformers import SentenceTransformer

from src.pipeline.json_extraction import DiagnosesModel, ICDsModel, extract_jsons
from src.pipeline.verifier import extract_and_map_diagnose_name
from src.pipeline.verifier_args import VerifierArgs
from src.utils import convert_codes_to_short_codes


def extract_diagnose_names_from_json(
        v_args: VerifierArgs,
        patients: pd.DataFrame,
        model: SentenceTransformer
) -> list[list[list[str]] | None]:
    """Extracts name from dict name/reason format.
    Maps predicted name to a potential diagnosis based on the primary complaint."""
    matched_names = []

    # Iterate over all primary complaints
    for primary_complaint, potential_diagnoses in v_args.potential_diagnoses.items():
        sub_df = patients[patients['Chief Complaint'] == primary_complaint]
        label_embeddings = model.encode(
            potential_diagnoses,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        for json in sub_df['v2_json'].tolist():
            if json is not None:
                matched_names.append(extract_and_map_diagnose_name(model, json, potential_diagnoses,
                                                                   label_embeddings))
            else:
                matched_names.append([])

    return matched_names


def extract_icd_names(v4_json: list, potential_icds: set = None) -> list:
    if v4_json == 'None' or not v4_json:
        return []

    codes = convert_codes_to_short_codes([codes['icd_code'] for codes in v4_json])
    if potential_icds:
        return [code for code in codes if code in potential_icds]
    else:
        return codes


def extract_json_and_pred_from_text(df: pd.DataFrame, v_args: VerifierArgs,
                                    mapping_model: SentenceTransformer, generation=False):
    df['v2_json'] = df['v2_text'].apply(lambda text: extract_jsons(text, DiagnosesModel))
    df['v4_json'] = df['v4_text'].apply(lambda text: extract_jsons(text, ICDsModel))
    df['v2_preds'] = extract_diagnose_names_from_json(v_args, df, mapping_model)
    df['v4_preds'] = df['v4_json'].apply(lambda json: extract_icd_names(json, v_args.potential_icds))

    if generation:
        print('v1')
        # df['v1_json'] = df['v1_text'].apply(lambda text: read_symptom_features(text, DiagnosesModel))
        print('get v1 preds')
        # df['v1_preds'] = df['v1_json'].map(get_symptom_vectors)
        print('v3')
        df['v3_json'] = df['v3_text'].apply(lambda text: extract_jsons(text, DiagnosesModel))
        df['v3_preds'] = extract_diagnose_names_from_json(v_args, df, mapping_model)



