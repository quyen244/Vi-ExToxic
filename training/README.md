# Vi-ExToxic SFT

This package trains a QLoRA adapter with Unsloth and TRL and evaluates it on a held-out CSV.

The default training source is `dataset_phase_1.csv` inside `data/archive.zip`.

The loader reads ZIP members in place and never extracts or modifies the archive.

The default student is `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`.

## Mixed precision on Kaggle GPUs

The trainer detects the active CUDA GPU before selecting its precision settings.

On BF16-capable GPUs it loads the model in BF16 and enables BF16 training.

On NVIDIA T4 (compute capability 7.5), it loads the model in FP16 and disables Trainer AMP/GradScaler.

This avoids the BF16-gradient unscale failure observed when FP16 GradScaler is enabled for this Unsloth QLoRA setup.


Pass `--model` to use another Hugging Face model ID or a local model directory.

## Comparison runs

Use the same model, train CSV, sequence length, random seed and SFT arguments for both targets.

`label-only` is the hard-label baseline and trains the model to return only `final_label`.

`distilled` trains on the teacher's explanation fields and `final_label`.

Both modes use the teacher-labeled training data in the archive.

The archive does not contain independent human annotations, so these runs are label-only SFT and teacher-label distillation, not a human-labeled baseline.

The held-out file contains 320 examples, including only one `Ambiguous/Noise` example and six `Identity-Based Hate` examples.

Treat the resulting Macro-F1 as exploratory because the held-out class support is very sparse.

## Local data validation

Create an isolated Python 3.12 environment before installing project training dependencies.

The dataset validation commands below use only the Python standard library and do not load a model or CUDA.

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m compileall -q training
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --target-mode label-only --output-dir outputs\label-only --validate-only
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --target-mode distilled --output-dir outputs\distilled --validate-only
```

To validate the holdout schema, select its ZIP member explicitly:

```powershell
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --zip-member labeled_test_dataset_phase_1.csv --target-mode distilled --output-dir outputs\validation --validate-only
```

An empty `input_emotion` is allowed and rendered in the prompt as `Không rõ`.

Distilled mode rejects missing or empty teacher explanation fields.

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

Run the label-only baseline and distilled SFT with matching arguments:

```python
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode label-only --output-dir /kaggle/working/outputs/label-only --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --seed 3407
!python -m training.train_sft --data /kaggle/input/datasets/quyenuit24/vi-extoxic/archive/dataset_phase_1.csv --target-mode distilled --output-dir /kaggle/working/outputs/distilled --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 --max-seq-length 2048 --seed 3407
```

Enable Kaggle Internet to download the default Hugging Face model, or pass a local model path with `--model`.

Model training artifacts are written under `/kaggle/working/outputs` for download.

### Merge and publish to Hugging Face

Add a Kaggle Notebook Secret named `HF_TOKEN` with a Hugging Face token that can write model repositories.

The publishing script merges the LoRA adapter into a 16-bit model and publishes it as a private Hugging Face model repository by default.

Keep the baseline and distilled models in separate repositories.

```python
!python -m training.publish_model --adapter /kaggle/working/outputs/label-only/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-label-only --target-mode label-only
!python -m training.publish_model --adapter /kaggle/working/outputs/distilled/adapter --repo-id YOUR_HF_USERNAME/vi-extoxic-distilled --target-mode distilled
```

Replace `YOUR_HF_USERNAME` with your Hugging Face account name.

Pass `--public` only if the model repository should be public.

See [the Kaggle guide](../kaggle_upload/README.md) for the complete upload, training, evaluation, publishing, and local download sequence.

On the local machine, authenticate with `hf auth login`, then download a merged model with `hf download YOUR_HF_USERNAME/vi-extoxic-label-only --local-dir models/label-only`.

The downloaded model can be connected to the local inference UI as the next integration step.

## Docker with an NVIDIA GPU

The root `Dockerfile` uses Unsloth's CUDA-ready image to keep its PyTorch, CUDA and Unsloth stack aligned.

Build from the repository root:

```powershell
docker build -t vi-extoxic-sft .
```

Run the baseline in PowerShell with the archive mounted read-only and outputs persisted on the host:

```powershell
docker run --rm --gpus all -v "${PWD}/data:/workspace/data:ro" -v "${PWD}/outputs:/workspace/outputs" vi-extoxic-sft --data /workspace/data/archive.zip --zip-member dataset_phase_1.csv --target-mode label-only --output-dir /workspace/outputs/label-only
```

Run the distilled target by replacing `label-only` in the target and output paths with `distilled`.

Docker GPU access requires Docker Desktop with its WSL 2 engine and a compatible NVIDIA driver on Windows.

## Resume training

Checkpoints are saved at the configured step interval under the selected output directory.

Pass `--resume-from-checkpoint latest` to resume from the newest checkpoint under that output directory.

```powershell
.\.venv\Scripts\python -m training.train_sft --data data/archive.zip --target-mode label-only --output-dir outputs\label-only --resume-from-checkpoint latest
```

Alternatively, pass an explicit checkpoint directory, such as `--resume-from-checkpoint outputs/label-only/checkpoint-200`.

Keep each target mode in a separate output directory so checkpoints and adapters cannot be mixed.

## Evaluate

Evaluate both adapters against the same held-out examples.

Macro-F1 averages the F1 score for all five labels, assigning zero to a label with no true or predicted examples.

The evaluator writes Macro-F1, accuracy, per-class precision/recall/F1/support, and a confusion matrix to `evaluation.json`.

It writes each input, gold label, parsed prediction, and raw generation to `predictions.csv`.

With Docker, use the same held-out ZIP member for both adapters:

```powershell
docker run --rm --gpus all --entrypoint python -v "${PWD}/data:/workspace/data:ro" -v "${PWD}/outputs:/workspace/outputs" vi-extoxic-sft -m training.evaluate_sft --adapter /workspace/outputs/label-only/adapter --data /workspace/data/archive.zip --zip-member labeled_test_dataset_phase_1.csv --target-mode label-only --output-dir /workspace/outputs/label-only/evaluation
docker run --rm --gpus all --entrypoint python -v "${PWD}/data:/workspace/data:ro" -v "${PWD}/outputs:/workspace/outputs" vi-extoxic-sft -m training.evaluate_sft --adapter /workspace/outputs/distilled/adapter --data /workspace/data/archive.zip --zip-member labeled_test_dataset_phase_1.csv --target-mode distilled --output-dir /workspace/outputs/distilled/evaluation
```

Compare the `macro_f1` values from the two `evaluation.json` files.
# Mixed precision on Kaggle GPUs

The trainer detects the active CUDA GPU before selecting its precision settings.

On BF16-capable GPUs it loads the model in BF16 and enables BF16 training.

On NVIDIA T4 (compute capability 7.5), it loads the model in FP16 and disables Trainer AMP/GradScaler.

This avoids the BF16-gradient unscale failure observed when FP16 GradScaler is enabled for this Unsloth QLoRA setup.
