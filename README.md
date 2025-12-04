## MeRLIn-DDx: Clinically Grounded Dataset for Diagnostic Reasoning

MeRLIn-DDx (Reasoning Learning for Interpretable Differential Diagnoses) is a framework and dataset designed to train and evaluate Large Language Models (LLMs) for **differential diagnosis (DDx)** by enforcing an explicit, verifiable reasoning process that mirrors clinical decision-making. This work reframes diagnosis prediction as a verifiable reasoning process, bridging symbolic validation with empirical performance.

The resulting dataset comprises approximately **9,000 synthetic reasoning paths** derived from approximately 3,000 real-world cases in the **MIMIC-IV** database. These paths logically connect initial chief complaints (primarily abdominal pain) to final discharge diagnoses, offering an interpretable and clinically validated resource for Supervised Fine-Tuning (SFT) of reasoning LLMs.

---

### Core Components

The MeRLIn-DDx Framework unifies data generation, reasoning evaluation, and dataset creation within a single, clinically grounded pipeline.

#### 1. Generator-Verifier Pipeline

The system utilizes a four-stage generator-verifier pipeline that replicates the stepwise diagnostic reasoning process clinicians follow, moving beyond direct text classification for ICD-10 codes. Each stage operates in an instruction-generation-verification loop, cross-checking outputs against medical ground truths from **WikiDoc** or MIMIC-IV.

The four stages are:

* **$V_1$: Symptom Extraction:** Extracts observable evidence from the patient admission note, validated against a WikiDoc-derived symptom schema.
* **$V_2$: Diagnosis Prediction:** Explores possible differential diagnoses, cross-checked against WikiDoc symptom-disease mappings to enforce medically valid hypothesis generation.
* **$V_3$: Lab-based Reranking:** Integrates objective laboratory data collected within 12 hours of admission to refine and rerank the diagnostic hypotheses, constraining the diagnostic space.
* **$V_4$: ICD Codes Prediction:** Maps all previous outputs to standardized **ICD-10 codes**, evaluated against MIMIC-IV discharge codes to ensure alignment with clinical documentation standards.

#### 2. MeRLIn-DDx Dataset

The dataset focuses on patients presenting with **abdominal pain** as the chief complaint, chosen for its frequency and diagnostic complexity.

* **Source Data:** Simulated admission notes from the MIMIC-IV dataset.
* **Scope:** 3,055 clinical notes and 9,165 verified reasoning chains.
* **Format:** The reasoning traces are converted into instruction-response pairs for Supervised Fine-Tuning (SFT).

---

### Setup and Requirements

The following infrastructure and configuration are essential for replicating the data generation process.

#### Data Requirements

* **MIMIC-IV:** Access to the MIMIC-IV database is required to obtain raw clinical notes and ground truth ICD discharge codes.
* **WikiDoc:** Used as the clinical knowledge base for structured symptom-disease mappings, serving as the ground truth for verification stages $V_1$, $V_2$, and $V_3$.

#### Infrastructure Requirements

The generation process utilizes large foundation models (Qwen3-32B, MedGemma-27B, Llama-3.1-70B Instruct) and is resource-intensive due to the iterative nature of generating traces per stage.

* **Kubernetes Cluster:** A **Kubernetes cluster** is necessary to manage and execute the instruction-generation-verification loops efficiently and scale the deployment of the server and client components.
* **WandB (Weights & Biases):** Used for experiment tracking, logging, and monitoring the SFT process and evaluation metrics.

---

### Scripts and Configuration

The `scripts` folder contains components for infrastructure management and initial data preparation for the generator-verifier pipeline.

| Script/File | Function                  | Description |
| :--- |:--------------------------| :--- |
| `build_docker` | **Containerization**      | Builds the **Docker image** necessary to run the MeRLIn-DDx client and server components, ensuring a standardized execution environment across the cluster. |
| `map_diagnoses` | **Data Preprocessing**    | Implements the logic to derive verifiable ground truths by mapping MIMIC-IV cases to WikiDoc disease entries and symptom profiles, which is essential for the verifier stages $V_1$, $V_2$, and $V_3$. |
| `run_from_config` | **Experiment Execution**  | Reads parameters from `server_client_config.yaml` to initialize and launch the deployed **server** (hosting the LLMs) and the **client** (managing the workflow). |
| `shutdown_from_config` | **Experiment Shutdown**   | Terminates the running client and server instances based on the current configuration, managing resource allocation. |
| `restart_client` | **Experiment Restart**    | Restarts the client process with new parameters (e.g., to adjust generation settings) without the need to restart the resource-heavy LLM server. |
| `server_client_config.yaml` | **Experiment Parameters** | Defines all critical parameters for the experiment, including model selection, verifier acceptance thresholds, iteration budgets, and deployment settings for the client and server. |

**Note:** Scripts for further preprocessing and supervised fine-tuning (SFT) are pending addition.