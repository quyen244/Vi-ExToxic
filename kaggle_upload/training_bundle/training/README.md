# Vi-ExToxic SFT

This package trains a QLoRA adapter with Unsloth and TRL and evaluates it on a held-out CSV.

The default training source is `dataset_phase_1.csv` inside `data/archive.zip`.

The loader reads ZIP members in place and never extracts or modifies the archive.

The default student is `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`.

## Student prompt and label definitions

The student prompt is built in `training/prompts.py` and follows the role, task, Vietnamese social-media expertise, and five label definitions in `notebooks/prompts/prompt_version_1.txt`.

The phase 1 CSV contains `confidence_score` as a teacher self-reported value.

There are four trained adapters grouped into three comparisons: the no-CoT baseline, CoT-distilled, and CoT prompt-language comparison (Vietnamese versus English).

`no-cot` predicts only `final_label`, making it the label-only baseline.

`cot` uses Vietnamese instructions to decode wording and slang, compare text with emotion, identify the target, and synthesize those steps in `thought_trace` before predicting `final_label` and numeric `confidence_score`.

`cot-en` uses English instructions for the same CoT task and the same teacher-generated Vietnamese completion targets as `cot`.

`cot` and `cot-en` are separate adapter runs so their held-out results compare prompt language under the same dataset, labels, target fields, and training settings.

`cot-distilled` uses Vietnamese instructions and returns the four structured reasoning fields (`semantic_decoding`, `slang_interpretation`, `contextual_conflict`, and `target`), synthesizes them in `thought_trace`, then predicts `final_label` and numeric `confidence_score`.

The five definitions are `Constructive/Clean` for harmless or positive content; `Implicit Toxicity` for indirect or sarcastic attacks; `Explicit Hostility` for direct insults, profanity, offensive language, or threats; `Identity-Based Hate` for attacks based on region, gender, religion, appearance, or body; and `Ambiguous/Noise` when the text is too unclear to interpret reliably.

## Mixed precision on Kaggle GPUs

The trainer detects the active CUDA GPU before selecting its precision settings.

On BF16-capable GPUs it loads the model in BF16 and enables BF16 training.

On NVIDIA T4 (compute capability 7.5), it loads the model in FP16 and disables Trainer AMP/GradScaler.

This avoids the BF16-gradient unscale failure observed when FP16 GradScaler is enabled for this Unsloth QLoRA setup.

Pass `--model` to use another Hugging Face model ID or a local model directory.

## Comparison runs

Use the same model, train CSV, sequence length, random seed and SFT arguments for all four adapter runs.

`no-cot` is the hard-label baseline and trains the model to return only `final_label`.

`cot` and `cot-en` train on the teacher's same `thought_trace`, label, and confidence.

`cot-distilled` adds the teacher's four structured reasoning fields to the CoT target.

The evaluation command checks the adapter's saved run metadata and rejects a target-mode or prompt-language mismatch.

Inference-only rare-class few-shot examples are stored in `training/few_shot_examples.json`.
The fixed three-shot set contains the user-provided Implicit Toxicity sample plus one Identity-Based Hate and one Ambiguous/Noise example from the training split.
Pass `--few-shot` to `training.evaluate_sft` for the no-CoT, Vietnamese CoT, or CoT-distilled few-shot runs.

All variants use the same teacher-labeled training data in the archive.

The archive does not contain independent human annotations, so these variants use teacher labels and rationales rather than human-verified supervision.

The held-out file contains 320 examples, including only one `Ambiguous/Noise` example and six `Identity-Based Hate` examples.

Treat the resulting Macro-F1 as exploratory because the held-out class support is very sparse.

## Local data validation

Create an isolated Python 3.12 environment before installing project training dependencies.

The dataset validation commands below use only the Python standard library and do not load a model or CUDA.

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m compileall -q training
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --target-mode no-cot --output-dir outputs\no-cot --validate-only
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --target-mode cot --output-dir outputs\cot --validate-only
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --target-mode cot-en --output-dir outputs\cot-en --validate-only
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --target-mode cot-distilled --output-dir outputs\cot-distilled --validate-only
```

To validate the holdout schema, select its ZIP member explicitly:

```powershell
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --zip-member labeled_test_dataset_phase_1.csv --target-mode cot-distilled --output-dir outputs\validation --validate-only
```

An empty `input_emotion` is allowed and rendered in the prompt as `Không rõ`.

CoT modes reject missing teacher rationale fields, nonnumeric confidence, or confidence outside [0, 1].

## Kaggle GPU

Attach `archive.zip` as a Kaggle input dataset and make the repository code available in the notebook working directory.

Select a GPU accelerator in notebook settings before installing dependencies.

For the attached `quyenuit24/vi-extoxic` dataset, install dependencies from its mounted path:

```python
!pip install -r /kaggle/input/datasets/quyenuit24/vi-extoxic/training_bundle/training/requirements.txt
```

Run from writable Kaggle working storage and add the read-only source package to `PYTHONPATH`.

This keeps model and compilation caches off the attached input dataset.

```python
%cd /kaggle/working
%env PYTHONPATH=/kaggle/input/datasets/quyenuit24/vi-extoxic/training_bundle
%env HF_HOME=/kaggle/working/.cache/huggingface
%env HF_HUB_CACHE=/kaggle/working/.cache/huggingface/hub
%env XDG_CACHE_HOME=/kaggle/working/.cache
!mkdir -p /kaggle/working/.cache/huggingface/hub /kaggle/working/unsloth_compiled_cache /kaggle/working/huggingface_tokenizers_cache
```

Run the four adapters in three experiment groups: no-CoT baseline, CoT-distilled, and the CoT prompt-language comparison.

```python
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode no-cot --output-dir /kaggle/working/outputs/no-cot --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --seed 3407
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode cot-distilled --output-dir /kaggle/working/outputs/cot-distilled --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --seed 3407
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode cot --output-dir /kaggle/working/outputs/cot --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --seed 3407
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode cot-en --output-dir /kaggle/working/outputs/cot-en --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --seed 3407
```

`cot` uses Vietnamese instructions and `cot-en` uses English instructions.
Both use the same Vietnamese teacher-generated output targets and record their prompt language and version in the adapter metadata.

Evaluate each adapter with its matching `--target-mode`.
Then repeat inference with `--few-shot` for the three Vietnamese modes; this adds the same three fixed demonstrations from `training/few_shot_examples.json` without changing the trained adapters.
The CoT-English adapter is evaluated without demonstrations so the CoT language comparison remains a no-shot comparison.

```python
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/no-cot/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode no-cot --max-seq-length 4096 --output-dir /kaggle/working/outputs/no-cot/evaluation
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/cot-distilled/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode cot-distilled --max-seq-length 4096 --output-dir /kaggle/working/outputs/cot-distilled/evaluation
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/cot/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode cot --max-seq-length 4096 --output-dir /kaggle/working/outputs/cot/evaluation
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/cot-en/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode cot-en --max-seq-length 4096 --output-dir /kaggle/working/outputs/cot-en/evaluation
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/no-cot/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode no-cot --few-shot --max-seq-length 4096 --output-dir /kaggle/working/outputs/no-cot/evaluation-few-shot
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/cot/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode cot --few-shot --max-seq-length 4096 --output-dir /kaggle/working/outputs/cot/evaluation-few-shot
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/cot-distilled/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode cot-distilled --few-shot --max-seq-length 4096 --output-dir /kaggle/working/outputs/cot-distilled/evaluation-few-shot
```

Use the same `--max-seq-length 4096` on every evaluation so all prompt treatments share the same context limit.
The evaluator records whether few-shot examples were used and verifies the adapter's training mode, prompt language, and prompt version.

Generate the formatted report after all seven evaluations:

```python
!python -m training.present_results --results-dir /kaggle/working/outputs --output /kaggle/working/outputs/comparison.md
```

Enable Kaggle Internet to download the default Hugging Face model, or pass a local model path with `--model`.

Model training artifacts are written under `/kaggle/working/outputs` for download.

### Merge and publish to Hugging Face

Add a Kaggle Notebook Secret named `HF_TOKEN` with a Hugging Face token that can write model repositories.

The publishing script merges the LoRA adapter into a 16-bit model and publishes it as a private Hugging Face model repository by default.

Publish each target mode to its own repository.

```python
!python -m training.publish_model --adapter /kaggle/working/outputs/no-cot/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-no-cot --target-mode no-cot
!python -m training.publish_model --adapter /kaggle/working/outputs/cot/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-cot --target-mode cot
!python -m training.publish_model --adapter /kaggle/working/outputs/cot-en/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-cot-en --target-mode cot-en
!python -m training.publish_model --adapter /kaggle/working/outputs/cot-distilled/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-cot-distilled --target-mode cot-distilled
```

Replace `YOUR_HF_USERNAME` with your Hugging Face account name.

Pass `--public` only if the model repository should be public.

See [the Kaggle guide](../kaggle_upload/README.md) for the complete upload, training, evaluation, publishing, and local download sequence.

On the local machine, authenticate with `hf auth login`, then download a merged model with `hf download YOUR_HF_USERNAME/vi-extoxic-no-cot --local-dir models/no-cot`.

The downloaded model can be connected to the local inference UI as the next integration step.

## Docker with an NVIDIA GPU

The root `Dockerfile` uses Unsloth's CUDA-ready image to keep its PyTorch, CUDA and Unsloth stack aligned.

Build from the repository root:

```powershell
docker build -t vi-extoxic-sft .
```

Run the baseline in PowerShell with the archive mounted read-only and outputs persisted on the host:

```powershell
docker run --rm --gpus all -v "${PWD}/data:/workspace/data:ro" -v "${PWD}/outputs:/workspace/outputs" vi-extoxic-sft --data /workspace/data/archive.zip --zip-member dataset_phase_1.csv --target-mode no-cot --output-dir /workspace/outputs/no-cot
```

Use `--target-mode cot`, `--target-mode cot-en`, or `--target-mode cot-distilled` and matching output directories for the other variants.

Docker GPU access requires Docker Desktop with its WSL 2 engine and a compatible NVIDIA driver on Windows.

## Resume training

Checkpoints are saved at the configured step interval under the selected output directory.

Pass `--resume-from-checkpoint latest` to resume from the newest checkpoint under that output directory.

```powershell
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --target-mode no-cot --output-dir outputs\no-cot --resume-from-checkpoint latest
```

Alternatively, pass an explicit checkpoint directory, such as `--resume-from-checkpoint outputs/no-cot/checkpoint-200`.

Keep each target mode in a separate output directory so checkpoints and adapters cannot be mixed.

## Evaluate

Evaluate all four adapters against the same held-out examples.

Macro-F1 averages the F1 score for all five labels, assigning zero to a label with no true or predicted examples.

The evaluator writes Macro-F1, accuracy, per-class precision/recall/F1/support, and a confusion matrix to `evaluation.json`.

It writes each input, gold label, parsed prediction, and raw generation to `predictions.csv`.

With Docker, use the same held-out ZIP member for every adapter:

```powershell
docker run --rm --gpus all --entrypoint python -v "${PWD}/data:/workspace/data:ro" -v "${PWD}/outputs:/workspace/outputs" vi-extoxic-sft -m training.evaluate_sft --adapter /workspace/outputs/no-cot/adapter --data /workspace/data/archive.zip --zip-member labeled_test_dataset_phase_1.csv --target-mode no-cot --output-dir /workspace/outputs/no-cot/evaluation
docker run --rm --gpus all --entrypoint python -v "${PWD}/data:/workspace/data:ro" -v "${PWD}/outputs:/workspace/outputs" vi-extoxic-sft -m training.evaluate_sft --adapter /workspace/outputs/cot/adapter --data /workspace/data/archive.zip --zip-member labeled_test_dataset_phase_1.csv --target-mode cot --output-dir /workspace/outputs/cot/evaluation
docker run --rm --gpus all --entrypoint python -v "${PWD}/data:/workspace/data:ro" -v "${PWD}/outputs:/workspace/outputs" vi-extoxic-sft -m training.evaluate_sft --adapter /workspace/outputs/cot-en/adapter --data /workspace/data/archive.zip --zip-member labeled_test_dataset_phase_1.csv --target-mode cot-en --output-dir /workspace/outputs/cot-en/evaluation
docker run --rm --gpus all --entrypoint python -v "${PWD}/data:/workspace/data:ro" -v "${PWD}/outputs:/workspace/outputs" vi-extoxic-sft -m training.evaluate_sft --adapter /workspace/outputs/cot-distilled/adapter --data /workspace/data/archive.zip --zip-member labeled_test_dataset_phase_1.csv --target-mode cot-distilled --output-dir /workspace/outputs/cot-distilled/evaluation
```

Compare the `macro_f1` values from the four `evaluation.json` files or generate a Markdown report with `python -m training.present_results --results-dir outputs --output outputs/comparison.md`.
# Mixed precision on Kaggle GPUs

The trainer detects the active CUDA GPU before selecting its precision settings.

On BF16-capable GPUs it loads the model in BF16 and enables BF16 training.

On NVIDIA T4 (compute capability 7.5), it loads the model in FP16 and disables Trainer AMP/GradScaler.

This avoids the BF16-gradient unscale failure observed when FP16 GradScaler is enabled for this Unsloth QLoRA setup.
