import pandas as pd
# import txtai
# from txtai import Embeddings

from src.exp_args import ExpArgs


SYMPTOM_FILES = {
    'abdominal pain': 'abdominal_pain.csv',
    'abdominal_pain': 'abdominal_pain.csv'
}


def map_pred_to_names(preds: list, names: list) -> list:
    def convert_pred_to_name(pred):
        return names[pred[0]]

    return [[convert_pred_to_name(item) for item in sublist] for sublist in preds]


def get_disease_df(chief_complaint: str) -> pd.DataFrame:
    """Load disease names for a given chief complaint. Expand by synonyms."""
    diagnose_path = f'data/medical_schemes/diagnoses/{SYMPTOM_FILES[chief_complaint]}'
    disease_df = pd.read_csv(diagnose_path)
    disease_names = []
    for _, row in disease_df.iterrows():
        disease_names.append({'Disease': row['Disease'], 'Name': row['Disease']})
        if pd.notna(row['Synonyms']):
            synonyms = row['Synonyms'].split('; ')
            for synonym in synonyms:
                disease_names.append({'Disease': row['Disease'], 'Name': synonym})
    return pd.DataFrame(disease_names)


def build_index(
        diagnoses_names: list,
        embedding_model='neuml/pubmedbert-base-embeddings',
) -> Embeddings:
    embeddings = txtai.Embeddings(path=embedding_model, content=False)
    embeddings.index(diagnoses_names)
    return embeddings


def get_query_strings(patients: pd.DataFrame, max_icds: int) -> str:
    """Generate a query string for the patient based on discharge diagnosis and ICD codes."""
    icd_desc = pd.read_csv('data/mimic-iv/icd_codes_9.csv')[['DIAGNOSIS CODE', 'LONG DESCRIPTION']]
    icd_desc = icd_desc.set_index('DIAGNOSIS CODE').to_dict()['LONG DESCRIPTION']
    icd10 = pd.read_csv('data/mimic-iv/icd_codes_10.csv')[['CODE', 'LONG DESCRIPTION']]
    icd_desc.update(icd10.set_index('CODE').to_dict()['LONG DESCRIPTION'])

    def get_query_string(patient: pd.Series) -> str:
        diagnosis = patient['Discharge Diagnosis']
        icd_codes = patient['ICD_CODES']
        icd_descriptions = [icd_desc[code] for code in icd_codes[:max_icds] if code in icd_desc]

        joined_icds = '\n'.join(icd_descriptions)
        query = f"Patient's discharge diagnosis: {diagnosis}.\nICD codes:\n{joined_icds}."

        return query

    return patients.apply(get_query_string, axis=1).tolist()


def get_disease_vector(args: ExpArgs, chief_complaint: str, disease_name: str) -> list:
    disease_df = pd.read_csv(f'{args.data_dir}/medical_schemes/diagnoses/{chief_complaint}.csv')
    disease_df.drop('Synonyms', axis=1, inplace=True)
    disease_df = disease_df[disease_df['Disease'] == disease_name]
    disease_df = disease_df.drop(['Disease', 'icd10_regex'], axis=1)
    return disease_df.values[0]


def add_mapped_primary_diagnoses(args: ExpArgs, file_name: str):
    """Load mimic patients, map primary diagnose -> top 10 wikidoc diagnose, store"""
    # Todo: load primary complaint dynamically
    exp_args = ExpArgs()
    file_path = f'{args.data_dir}/reasoning/abdominal_pain/{file_name}'
    patients = pd.read_parquet(file_path)
    patients.set_index('hadm_id', inplace=True)

    chief_complaints = patients['Chief Complaint'].unique()

    for chief_complaint in chief_complaints:
        # Filter for chief complaint and load related diagnose names
        sub_patients = patients[patients['Chief Complaint'] == chief_complaint]

        # Create index with diagnose names
        disease_df = get_disease_df(chief_complaint)
        index = build_index(disease_df['Name'].tolist())
        disease_names = disease_df['Disease'].tolist()

        # Predict 10 diagnoses based on discharge diagnosis string
        predictions = index.batchsearch(get_query_strings(sub_patients, 2), 10)
        predictions = map_pred_to_names(predictions, disease_names)
        diseases = [pred[0] for pred in predictions]
        d_vectors = [get_disease_vector(exp_args, chief_complaint, disease) for disease in diseases]
        predictions = pd.Series(predictions, sub_patients.index)
        patients.loc[sub_patients.index, 'wikidoc_diagnoses'] = pd.Series(predictions, sub_patients.index)
        patients.loc[sub_patients.index, 'disease'] = pd.Series(diseases, sub_patients.index)
        patients.loc[sub_patients.index, 'disease_vector'] = pd.Series(d_vectors, sub_patients.index)

    patients.to_parquet(file_path)

    return patients
