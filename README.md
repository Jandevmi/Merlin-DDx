# Merlin-DDx  
A knowledge-augmented differential diagnosis system combining structured medical knowledge and large language model inference.

## Purpose & Scope  
This document introduces Merlin-DDx, a platform for medical differential diagnosis that takes patient presentations (chief complaints and associated symptoms) and outputs ranked lists of possible diagnoses. The system integrates structured medical knowledge with neural reasoning to improve clinical accuracy and interpretability.

For detailed subsystems, see:  
- Medical Knowledge Base  
- Model Infrastructure  
- Evaluation Framework  
- Clinical Reasoning Pipeline

## System Description  
Merlin-DDx uses a hybrid symbolic-neural architecture:  
- **Symbolic component**: structured medical knowledge (diagnosis databases, symptom schemas, ICD code mappings)  
- **Neural component**: large language models (LLMs, 0.6B to 70B parameters) hosted via vLLM  
- **Integration layer**: orchestration logic that retrieves knowledge, constructs prompts, invokes the LLM, and validates output  

This approach constrains and guides model reasoning to improve accuracy and interpretability compared to purely end-to-end LLM approaches. :contentReference[oaicite:0]{index=0}

## Key Capabilities  
| Capability              | Description                                                          |
|--------------------------|-----------------------------------------------------------------------|
| Multi-symptom coverage   | Handles 8 primary symptom presentations (387+ conditions) :contentReference[oaicite:1]{index=1} |
| Structured reasoning     | Uses tri-state feature encoding (-1 / 0 / 1) across clinical features :contentReference[oaicite:2]{index=2} |
| ICD code validation      | Maps diagnoses to ICD-9/10 codes via regex matching :contentReference[oaicite:3]{index=3} |
| Multiple model support   | Evaluates models from 0.6B to 70B parameters :contentReference[oaicite:4]{index=4} |
| Guided generation        | Constrains outputs to valid JSON structures :contentReference[oaicite:5]{index=5} |
| Cross-validation eval    | Rigorous testing across folds and out-of-domain datasets :contentReference[oaicite:6]{index=6} |

## System Architecture Overview  

### Core Components  
**Medical Knowledge Base**  
- Diagnosis databases: `data/medical_schemes/diagnoses/*.csv` — lists conditions per symptom category with feature profiles. :contentReference[oaicite:7]{index=7}  
- Symptom schemas: `data/medical_schemes/symptoms/*.yaml` — defines clinical features for data collection. :contentReference[oaicite:8]{index=8}  
- ICD code maps: `data/mimic-iv/icd_codes_*.csv` — maps diseases to standard ICD codes. :contentReference[oaicite:9]{index=9}  
- Coverage examples:  
  - Abdominal Pain: 69 conditions, 24 features :contentReference[oaicite:10]{index=10}  
  - Chest Pain: 49 conditions, 25 features :contentReference[oaicite:11]{index=11}  
  - Dyspnea: 77 conditions, 21 features :contentReference[oaicite:12]{index=12}  
  - … etc.  

**vLLM Client-Server Architecture**  
- Server container: built from `vllm/vllm-openai:nightly`, adds the `outlines` library for JSON schema enforcement, exposes OpenAI-compatible API, supports GPU. :contentReference[oaicite:13]{index=13}  
- Client container: CUDA 12.2 base with cuDNN 8, mounts the medical knowledge base, orchestrates prompt construction → server call → output validation. :contentReference[oaicite:14]{index=14}  
- Deployed on Kubernetes: separate pods for server & client, medical knowledge base mounted as volume, environment variables configure model paths & endpoints, uses WandB for experiment tracking. :contentReference[oaicite:15]{index=15}  

## Diagnostic Reasoning Pipeline  
The end-to-end flow from patient presentation to ranked diagnoses includes:  
- `load_schemes_and_labelspace()` — loads diagnosis CSVs and symptom schemas. :contentReference[oaicite:16]{index=16}  
- `PromptArgs` — manages prompt configuration and templates. :contentReference[oaicite:17]{index=17}  
- `extract_json_and_pred_from_text()` — parses LLM output and extracts predictions. :contentReference[oaicite:18]{index=18}  
- `load_sbert_model()` — loads sentence-BERT for semantic matching. :contentReference[oaicite:19]{index=19}  
- `calculate_icd_metrics()` — computes ICD code accuracy metrics. :contentReference[oaicite:20]{index=20}  
- `calculate_disease_metrics()` — computes Mean Reciprocal Rank (MRR), Variant Reciprocal Rank (VRR). :contentReference[oaicite:21]{index=21}  
- `convert_codes_to_short_codes()` — normalizes ICD codes. :contentReference[oaicite:22]{index=22}  
- `init_wandb()` — initializes WandB experiment tracking. :contentReference[oaicite:23]{index=23}  

## Model Evaluation Framework  
**Evaluated Models**  
- Qwen3: 0.6B, 8B, 14B, 32B — Base, Guided (-G), LoRA fine-tuned, MIMIC-adapted. :contentReference[oaicite:24]{index=24}  
- Llama 3.3: 70B — Base, Guided (-G). :contentReference[oaicite:25]{index=25}  
- MedGemma: 27B — Base, Guided (-G). :contentReference[oaicite:26]{index=26}  
- MedReason: 8B — Base, Guided (-G). :contentReference[oaicite:27]{index=27}  

**Evaluation Metrics**  
- V1_CosSim: cosine similarity between predicted & true symptom-disease vectors. :contentReference[oaicite:28]{index=28}  
- V2_MRR: Mean Reciprocal Rank of correct diagnosis in ranked list. :contentReference[oaicite:29]{index=29}  
- V3_VRR: Variant Reciprocal Rank. :contentReference[oaicite:30]{index=30}  
- ICD Accuracy: precision/recall for ICD code predictions. :contentReference[oaicite:31]{index=31}  
- Valid JSON %: percentage of outputs with valid JSON structure. :contentReference[oaicite:32]{index=32}  

**Test Scenarios**  
- In-domain: dataset for primary symptom category (e.g., abdominal pain). :contentReference[oaicite:33]{index=33}  
- Out-of-domain (OOD): dataset of 700 patients across all 8 symptom categories to test generalization. :contentReference[oaicite:34]{index=34}  
- 3-fold cross-validation per model, aggregate results, generate tables of mean ± standard deviation. :contentReference[oaicite:35]{index=35}  

## Data Encoding System  
Merlin-DDx uses a **tri-state encoding system** for clinical features:  
- `-1`: feature is absent or explicitly negative. 
- `0`: feature is neutral, unknown, or not applicable. 
- `1`: feature is present or explicitly positive. 

**Example (Acute appendicitis without perforation):**  
- RLQ pain: 1 :
- Fever: 1 
- Nausea/vomiting: 1
- Guarding: 1 
- Rebound tenderness: 1
- Hypoactive bowel sounds: 1 
- Diarrhea: 0 
- Jaundice: -1 

This encoding enables:  
- Precise clinical characterization.  
- Pattern-matching between patient data and disease profiles.  
- Semantic similarity calculations using cosine similarity.  
- Interpretable reasoning that clinicians can verify.

## Deployment Architecture  
Merlin-DDx is designed for Kubernetes deployment with two primary containers:  
- **vLLM Server**: `vllm/vllm-openai:nightly` image; requires P100/A100 for model hosting. 
- **vLLM Client**: NVIDIA CUDA 12.2 base; handles reasoning orchestration. 

## Summary  
Merlin-DDx implements a knowledge-augmented approach to medical differential diagnosis by combining:  
- Structured medical knowledge (8 symptom-specific diagnosis databases, 387+ conditions, standardized feature schemas). :contentReference[oaicite:54]{index=54}  
- Modern LLM infrastructure (vLLM client-server architecture with GPU acceleration). :contentReference[oaicite:55]{index=55}  
- Rigorous evaluation (cross-validation framework comparing multiple model families with 5+ metrics). :contentReference[oaicite:56]{index=56}  
- Clinical validation (ICD code mapping and tri-state encoding for interpretable reasoning). :contentReference[oaicite:57]{index=57}  

Its key innovation is the **hybrid symbolic-neural design**: structured medical schemas constrain and guide LLM outputs, improving both accuracy and interpretability over purely end-to-end approaches. :contentReference[oaicite:58]{index=58}  
