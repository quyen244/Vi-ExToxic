# Vi-ExToxic Kaggle Upload and Run

This folder is the Kaggle Dataset upload bundle for `quyenuit24/vi-extoxic`.

It contains the phase 1 archive and the complete phase 4 training package.

## 1. Prepare Kaggle authentication

Install the current Kaggle CLI in the project Python 3.12 environment and sign in with Kaggle's browser OAuth flow.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade kaggle
.\.venv\Scripts\kaggle.exe auth login
```

If you do not have the project virtual environment, create it first with `py -3.12 -m venv .venv`.

Do not paste an API token into source files or this chat.

## 2. Set the dataset owner in metadata

The metadata uses the Kaggle dataset ID `quyenuit24/vi-extoxic`.

If you upload under a different Kaggle account or dataset name, update the ID before uploading:

```powershell
$KaggleUsername = Read-Host "Kaggle username"
$MetadataPath = ".\kaggle_upload\dataset-metadata.json"
$Metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
$Metadata.id = "$KaggleUsername/vi-extoxic"
$Metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $MetadataPath -Encoding utf8
```

The license is marked `unknown` because the archive's redistribution license has not been established.

Keep the dataset private unless you have confirmed that you have the right to redistribute the source data.

## 3. Upload the private dataset

From the repository root, create the Kaggle Dataset.

Kaggle keeps newly created datasets private unless `--public` is specified.

If you downloaded `dist/vi-extoxic-kaggle-upload.zip`, extract it once and upload the extracted folder contents.

Do not put a `training_bundle.zip` beside the `training_bundle/` folder.

Kaggle extracts ZIP files into folders, so including both creates a `training_bundle` directory collision.

```powershell
.\.venv\Scripts\kaggle.exe datasets create -p .\kaggle_upload -t -r skip
```

If `quyenuit24/vi-extoxic` already exists, publish a new dataset version instead:

```powershell
.\.venv\Scripts\kaggle.exe datasets version -p .\kaggle_upload -m "Add no-CoT and CoT SFT variants" -r skip
```

The dataset includes `archive.zip`, the unpacked `training_bundle/training/` package, and the README and checksum files.

The package does not upload model weights or training outputs.

## 4. Create a Kaggle Notebook

Attach the new private dataset to a notebook and select a GPU accelerator.

Enable Notebook Internet so the runtime can download the default Hugging Face model.

The attached dataset is available at `/kaggle/input/datasets/quyenuit24/vi-extoxic`.

```python
!pip install -r /kaggle/input/datasets/quyenuit24/vi-extoxic/training_bundle/training/requirements.txt
```

Run from writable Kaggle working storage and put the read-only source package on `PYTHONPATH`.

Keep Hugging Face and Unsloth caches under `/kaggle/working`, not under the attached input dataset.

```python
%cd /kaggle/working
%env PYTHONPATH=/kaggle/input/datasets/quyenuit24/vi-extoxic/training_bundle
%env HF_HOME=/kaggle/working/.cache/huggingface
%env HF_HUB_CACHE=/kaggle/working/.cache/huggingface/hub
%env XDG_CACHE_HOME=/kaggle/working/.cache
!mkdir -p /kaggle/working/.cache/huggingface/hub /kaggle/working/unsloth_compiled_cache /kaggle/working/huggingface_tokenizers_cache
```

Validate the training and holdout CSVs before downloading model weights:

```python
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode no-cot --output-dir /kaggle/working/outputs/no-cot --validate-only
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode cot --output-dir /kaggle/working/outputs/cot --validate-only
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode cot-distilled --output-dir /kaggle/working/outputs/cot-distilled --validate-only
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode cot-distilled --output-dir /kaggle/working/outputs/holdout-validation --validate-only
```

## 5. Run all three SFT variants

These commands use the same student, data, seed, sequence length, and optimizer settings.

Run all three in the same notebook session when time allows.

Each process releases GPU memory when it exits.

```python
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode no-cot --output-dir /kaggle/working/outputs/no-cot --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --learning-rate 2e-4 --seed 3407
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode cot --output-dir /kaggle/working/outputs/cot --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --learning-rate 2e-4 --seed 3407
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode cot-distilled --output-dir /kaggle/working/outputs/cot-distilled --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --learning-rate 2e-4 --seed 3407
```

The default model is `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`.

Pass `--model <HF_MODEL_ID>` to each command if you select another model.

## 6. Evaluate all three variants on the shared holdout

```python
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/no-cot/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode no-cot --output-dir /kaggle/working/outputs/no-cot/evaluation
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/cot/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode cot --output-dir /kaggle/working/outputs/cot/evaluation
!python -m training.evaluate_sft --adapter /kaggle/working/outputs/cot-distilled/adapter --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/labeled_test_dataset_phase_1.csv --target-mode cot-distilled --output-dir /kaggle/working/outputs/cot-distilled/evaluation
```

Each evaluation directory contains `evaluation.json` and `predictions.csv`.

The holdout labels were produced by the teacher, and the rarest class has one example.

Treat Macro-F1 as exploratory and include per-class support in any report.

## 7. Merge the LoRA adapter and publish it to Hugging Face

The publish script merges the saved LoRA adapter into a 16-bit model and uploads the merged model to Hugging Face Hub.

In Kaggle, add a Notebook Secret named `HF_TOKEN` containing a Hugging Face token with write access to model repositories.

Publish each target mode to a separate repository for clear comparison.

The repositories are private by default; add `--public` only when you intend to publish the model publicly.

```python
!python -m training.publish_model --adapter /kaggle/working/outputs/no-cot/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-no-cot --target-mode no-cot
!python -m training.publish_model --adapter /kaggle/working/outputs/cot/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-cot --target-mode cot
!python -m training.publish_model --adapter /kaggle/working/outputs/cot-distilled/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-cot-distilled --target-mode cot-distilled
```

Replace `YOUR_HF_USERNAME` with your Hugging Face account name.

The default base model is `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`.

Pass `--base-model <HF_MODEL_ID>` if training used a different base model.

The script reads the token from the Kaggle Secret `HF_TOKEN`, checks authentication, creates the private model repository, merges the adapter, uploads it, and adds a model card.

It does not write the token into the notebook or the repository.

## 8. Download the published model for local inference

Install the project training dependencies in the local virtual environment, or install `huggingface-hub` separately.

For a private repository, authenticate locally with `hf auth login` before downloading.

Use `hf auth login` and paste the token into its interactive prompt; do not put it in a shell command or source file.

Download the merged models you want to compare into the ignored `models/` directory:

```powershell
hf download YOUR_HF_USERNAME/vi-extoxic-no-cot --local-dir models/no-cot
hf download YOUR_HF_USERNAME/vi-extoxic-cot-distilled --local-dir models/cot-distilled
```

Those local model folders can then be loaded by the project's inference UI when that local testing flow is implemented.

## 9. Preserve the results

Kaggle working storage is temporary.

Create one archive of adapters, run metadata, and evaluation outputs before the session ends:

```python
!cd /kaggle/working && zip -qr vi-extoxic-results.zip outputs
```

Save a notebook version with output and download `vi-extoxic-results.zip` from the notebook's output files.

If all three runs do not fit in one session, download or save each completed run before ending its session.

## Resume a run

If a checkpoint exists in the same Kaggle working session or has been restored to the same output directory, resume with `--resume-from-checkpoint latest`.

```python
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode no-cot --output-dir /kaggle/working/outputs/no-cot --resume-from-checkpoint latest --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --learning-rate 2e-4 --seed 3407
```

Use separate output directories for `cot` and `cot-distilled` runs.
