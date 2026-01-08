# Copilot / AI agent usage notes for DeMoE

Summary
- This repo implements DeMoE, a Mixture-of-Experts decoder for image deblurring. Core network code lives in `archs/` (notably `archs/DeMoE.py`, `archs/moeblocks.py`, `archs/arch_model.py`). Training and inference entrypoints are `finetune.py` and `inference.py`.

Quick run (inference)
- The repository provides `inference.sh` which runs a torchrun command invoking `inference.py`. Model weights expected under `models/` (default `models/DeMoE.pt`).
- `inference.py` CLI: `-p/--config` (YAML), `-c/--checkpoints_path`, `-i/--inp_path`, `-t/--task` (choices: `auto, defocus, global_motion, synth_global_motion, local_motion, low_light`).

Quick run (training)
- Use `finetune.py` with a YAML config: `python finetune.py -p options/train/DRMI_finetune.yml -c ./models/DeMoE.pt`.
- For multi-GPU run use `torchrun` / `torch.distributed` environment variables (scripts use `RANK` / `WORLD_SIZE`). `finetune.py` supports either being launched with `torchrun` or falling back to single-process init file.

Important files & locations
- Network implementations: [archs/DeMoE.py](archs/DeMoE.py#L1)
- MoE internals: [archs/moeblocks.py](archs/moeblocks.py#L1)
- Model factory: `archs/__init__.py` (exports `create_model`) used by `finetune.py` and `inference.py`.
- CLI config loader: [options/options.py](options/options.py#L1) — YAML -> OrderedDict parser used across scripts.
- Example configs: `options/inference/DeMoE.yml`, `options/train/DRMI_finetune.yml`.
- Data folders: `data/` and `data/DRMI_dataset/` — training expects paired images (meta + gt).
- Checkpoints: default `models/DeMoE.pt`; training saves to `experiments/<name>/models/{latest.pt,best.pt}`.

Model / checkpoint conventions
- Checkpoints are Python pickles (torch.save) storing a dict with at least a `params` key containing the `state_dict`. Example saved keys seen in code: `params`, `epoch`, `optimizer`, `best_psnr`.
- `finetune.py` contains `_adapt_state_dict_keys_for_model` to convert between `module.`-prefixed keys and non-prefixed keys — respect this when loading or altering checkpoints.

Data & Dataset conventions
- `finetune.py` defines `PairedImageFolderDataset` which expects two parallel folders: `meta_dir` (input/blurry) and `gt_dir` (ground truth) — filenames must match.
- Cropping/augmentation behavior controlled by YAML `train` options: `patch_size`, `augment`, `batch_size`, `num_workers`.

Runtime & distributed patterns
- Scripts set `CUDA_VISIBLE_DEVICES="0"` by default; be explicit when changing GPUs. `finetune.py` also sets `PL_TORCH_DISTRIBUTED_BACKEND=gloo` as a safety default.
- `finetune.py` supports being launched either under `torchrun` (environment contains `RANK`/`WORLD_SIZE`) or as a single-process with an init file. `inference.py` calls `dist.init_process_group('nccl')` and expects it to be launched via `torchrun` (it accesses `LOCAL_RANK`).
- When running distributed, model is wrapped into a DistributedDataParallel-like wrapper (`model.module` is referenced in `inference.py`). Be careful when calling `model.forward` vs `model.module.forward` depending on wrapping.

API & internal contracts
- Model forward: call as `model(input, task=...)` and expect a dict: keys include `output` (tensor), `pred_labels` (class weights), `weights`, and `bin_counts`.
- Task selection: `archs/DeMoE.py` defines `TASKS` mapping; passing `task!='auto'` forces the expert-selection weights to the mapped vector.

Editing guidelines & common pitfalls
- If changing checkpoint key names or wrapping, update `_adapt_state_dict_keys_for_model` in `finetune.py` and loader logic in `inference.py` accordingly.
- Inference pads inputs to multiples of `padder_size` (see `DeMoE.check_image_size`) — any change to encoder/decoder downsampling must consider padding logic.
- YAML -> runtime mapping uses plain `parse(opt_path)` in `options/options.py`. Avoid dynamic runtime-only config changes that aren't reflected in YAML (helps reproducibility).

Examples (useful snippets)
- Load and run inference on folder (single-GPU): edit `inference.sh` or run:
```
torchrun --nproc_per_node=1 inference.py -p options/inference/DeMoE.yml -c ./models/DeMoE.pt -i ./images/inputs -t auto
```
- Train (single GPU):
```
python finetune.py -p options/train/DRMI_finetune.yml -c ./models/DeMoE.pt
```

When to ask for clarification
- If you need to change model IO (checkpoint layout, state_dict keys), confirm whether external downstream tools expect the existing `{params: ...}` shape.
- If modifying distributed init (switching backends or multi-node), ask for the target cluster details (GPU count, NCCL availability).

If anything above is unclear or you want deeper guidance (e.g., adding a new dataset loader, changing expert routing), tell me which area to expand.
