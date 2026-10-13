# Phase 4 Training Contract

## Outcome and boundary

Prepare reproducible LoRA supervised fine-tuning for the existing Vietnamese teacher-labeled dataset, with a label-only baseline and an explanation-distilled student run.

The pipeline consumes the phase 1 archive and its labeled holdout, writes adapters and evaluation artifacts outside source data, and can run on a Kaggle GPU.

Phases 2 and 3 are skipped as requested.

## Confirmed inputs and constraints

- Source archive: `data/archive.zip`.
- Preferred training CSV in the archive: `dataset_phase_1.csv` (2,774 rows, one duplicate `input_text`).
- The typo-named `blanced_train_dataset_phase_1.csv.csv` has 3,810 rows but only 2,476 unique input texts, so it is not used as the default training source.
- Labeled test CSV in the archive: `labeled_test_dataset_phase_1.csv` (320 rows).
- Input fields: `input_text` and `input_emotion`.
- Teacher targets: four explanation fields and `final_label`.
- Existing code and docs point to a Qwen 2.5 3B instruct student, which is the default unless repository inspection finds a stronger blocker.
- The local GPU has 6 GB VRAM; full SFT execution is intended for Kaggle GPU. Local Docker is for environment and data-path validation, not a promise of fitting the 3B training workload on this GPU.
- The held-out set is highly imbalanced: `Ambiguous/Noise` has one example and `Identity-Based Hate` has six, so Macro-F1 must be reported with per-class support and interpreted cautiously.
- The held-out reference labels are teacher-generated; the skipped verification and cultural-review stages provide no independent gold-label evidence.

## Data and model relationship

```mermaid
flowchart LR
    A[data/archive.zip] --> B[Validate CSV schema and rows]
    B --> C[Label-only prompt/target]
    B --> D[Explanation-distilled prompt/target]
    C --> E[Same student and train configuration]
    D --> E
    E --> F[Unsloth + TRL SFTTrainer]
    F --> G[LoRA adapters and run metadata]
    G --> H[Shared labeled holdout evaluation]
    H --> I[Macro-F1, per-class metrics, predictions]
```

## Run lifecycle

```mermaid
stateDiagram-v2
    [*] --> Configured
    Configured --> DataValidated: schema and labels valid
    Configured --> Failed: invalid config or source
    DataValidated --> Training: model and GPU loaded
    DataValidated --> Failed: insufficient device/dependency
    Training --> Checkpointed: checkpoint interval reached
    Checkpointed --> Training: resume
    Training --> AdapterSaved: training completes
    AdapterSaved --> Evaluated: holdout inference completes
    Evaluated --> Complete: artifacts and metrics written
    Training --> Failed: unrecoverable training error
    Failed --> Configured: user fixes input/config and reruns
```

## Decisions and edge cases

- Baseline means label-only SFT using `final_label`; distilled SFT uses the teacher explanations plus `final_label`.
- The baseline-versus-distilled comparison is exploratory response-target ablation, not a claim that KD beats human-grounded supervised learning.
- Both variants use the same training rows, held-out rows, model family, seed, and training configuration so the comparison isolates the output-target style.
- The archive is read in place; setup must not extract over or rewrite the user's source data.
- Invalid/missing required columns and empty inputs fail before downloading a model.
- Invalid target labels fail with row context instead of being silently mapped.
- Checkpoints and final outputs live under an explicit output directory and can be resumed where supported by TRL.
- Evaluation must report per-class support because the held-out data has sparse classes.
- Full training is not an acceptance check on the local 6 GB GPU; the local container must at least build and expose the CLI/data validation path, while GPU training is intended for Kaggle.

## Acceptance criteria and delivery tasks

- [x] A CLI validates both CSVs inside the archive without modifying them.
- [x] Label-only and explanation-distilled targets render from the same row schema.
- [ ] An Unsloth `SFTTrainer` run saves a LoRA adapter, tokenizer, checkpoints, config, and logs.
- [ ] The same held-out set can be evaluated and produces Macro-F1 plus per-class support for each run.
- [x] Kaggle instructions show inputs, GPU setup, artifact persistence, and resume usage.
- [x] Docker provides a documented local validation/training environment with explicit GPU requirements.
- [x] README and architecture documentation describe only implemented behavior and identify current result values as pending.

The source/data validation criteria passed under the project-local Python 3.12 virtual environment.

The Docker image build is not yet verified because this machine has the Docker CLI but no running Docker Engine.

Model training and holdout inference remain pending because the local 6 GB GPU is below the intended training target and no Kaggle run has been performed.
