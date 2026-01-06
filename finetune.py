import argparse
import math
import os
import random
import tempfile
from pathlib import Path

from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from tqdm import tqdm

from archs import create_model
from options.options import parse

import os

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # 即使意外触发分布式，也用兼容性最好的后端
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 限制仅使用第1张GPU


class PairedImageFolderDataset(Dataset):
    def __init__(
        self,
        meta_dir: str,
        gt_dir: str,
        patch_size: int = 256,
        is_train: bool = True,
        augment: bool = True,
    ):
        self.meta_dir = Path(meta_dir)
        self.gt_dir = Path(gt_dir)
        self.patch_size = int(patch_size)
        self.is_train = bool(is_train)
        self.augment = bool(augment) and self.is_train

        self.to_tensor = transforms.ToTensor()

        meta_files = sorted([p for p in self.meta_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        pairs = []
        for mp in meta_files:
            gp = self.gt_dir / mp.name
            if gp.is_file():
                pairs.append((mp, gp))
        if len(pairs) == 0:
            raise ValueError(f"No paired images found. meta_dir={self.meta_dir} gt_dir={self.gt_dir}")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _pad_to_min_size(img: torch.Tensor, min_h: int, min_w: int):
        _, h, w = img.shape
        pad_h = max(0, min_h - h)
        pad_w = max(0, min_w - w)
        if pad_h == 0 and pad_w == 0:
            return img
        x = img.unsqueeze(0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x.squeeze(0)

    def _random_crop_pair(self, meta: torch.Tensor, gt: torch.Tensor):
        if self.patch_size <= 0:
            return meta, gt

        meta = self._pad_to_min_size(meta, self.patch_size, self.patch_size)
        gt = self._pad_to_min_size(gt, self.patch_size, self.patch_size)

        _, h, w = meta.shape
        if h == self.patch_size and w == self.patch_size:
            return meta, gt

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)

        meta = meta[:, top : top + self.patch_size, left : left + self.patch_size]
        gt = gt[:, top : top + self.patch_size, left : left + self.patch_size]
        return meta, gt

    def _augment_pair(self, meta: torch.Tensor, gt: torch.Tensor):
        if not self.augment:
            return meta, gt

        if random.random() < 0.5:
            meta = torch.flip(meta, dims=[2])
            gt = torch.flip(gt, dims=[2])
        if random.random() < 0.5:
            meta = torch.flip(meta, dims=[1])
            gt = torch.flip(gt, dims=[1])
        if random.random() < 0.5:
            meta = meta.transpose(1, 2)
            gt = gt.transpose(1, 2)
        return meta, gt

    def __getitem__(self, idx: int):
        meta_path, gt_path = self.pairs[idx]

        meta_img = Image.open(meta_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        meta = self.to_tensor(meta_img)
        gt = self.to_tensor(gt_img)

        if meta.shape != gt.shape:
            raise ValueError(f"Mismatched shapes for {meta_path.name}: meta={meta.shape}, gt={gt.shape}")

        if self.is_train:
            meta, gt = self._random_crop_pair(meta, gt)
            meta, gt = self._augment_pair(meta, gt)

        return {
            "meta": meta,
            "gt": gt,
            "name": meta_path.name,
        }


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return t
    rt = t.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / dist.get_world_size()
    return rt


@torch.no_grad()
def evaluate(model, loader, device, task: str = "auto", amp: bool = True):
    model.eval()

    total_psnr = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)

    pbar = loader
    if is_main_process():
        pbar = tqdm(loader, desc="val", leave=False)

    for batch in pbar:
        meta = batch["meta"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            out = model(meta, task=task)["output"]

        out = torch.clamp(out, 0.0, 1.0)
        mse = F.mse_loss(out, gt, reduction="none").mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))

        total_psnr += psnr.sum()
        total_count += torch.tensor(float(psnr.numel()), device=device)

    total_psnr = reduce_mean(total_psnr)
    total_count = reduce_mean(total_count)

    model.train()
    return (total_psnr / torch.clamp(total_count, min=1.0)).item()


def load_checkpoint_params(path: str, device: torch.device):
    ckpt = torch.load(path, map_location={"cuda:0": str(device)} if device.type == "cuda" else "cpu", weights_only=False)
    if isinstance(ckpt, dict) and "params" in ckpt:
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _adapt_state_dict_keys_for_model(model: torch.nn.Module, state_dict: dict) -> dict:
    model_keys = list(model.state_dict().keys())
    if len(model_keys) == 0:
        return state_dict

    sd_keys = list(state_dict.keys())
    if len(sd_keys) == 0:
        return state_dict

    model_has_module = model_keys[0].startswith("module.")
    sd_has_module = sd_keys[0].startswith("module.")

    if sd_has_module and not model_has_module:
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    if (not sd_has_module) and model_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="DeMoE finetuning")
    parser.add_argument("-p", "--config", type=str, required=True, help="Train config yaml")
    parser.add_argument("-c", "--pretrained", type=str, default="./models/DeMoE.pt", help="Pretrained checkpoint")
    parser.add_argument("--resume", type=str, default="", help="Resume checkpoint")
    parser.add_argument("--master_port", type=int, default=12547)
    args = parser.parse_args()

    opt = parse(args.config)

    launched_with_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    backend = "nccl" if (torch.cuda.is_available() and os.name != "nt") else "gloo"

    if launched_with_torchrun:
        dist.init_process_group(backend=backend)
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        global_rank = 0
        local_rank = 0
        init_file = Path(tempfile.gettempdir()) / f"demoe_ddp_{os.getpid()}_{args.master_port}.init"
        init_file.parent.mkdir(parents=True, exist_ok=True)
        dist.init_process_group(
            backend="nccl",
            init_method=init_file.as_uri(),
            rank=0,
            world_size=1,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    seed = int(opt.get("seed", 42)) + global_rank
    setup_seed(seed)

    model, _, _ = create_model(opt["network"], local_rank=local_rank, global_rank=global_rank)

    if args.pretrained:
        pretrained = load_checkpoint_params(args.pretrained, device)
        params = _adapt_state_dict_keys_for_model(model, pretrained["params"])
        model.load_state_dict(params, strict=True)
        if is_main_process():
            print(f"Loaded pretrained: {args.pretrained}")

    start_epoch = 0
    best_psnr = -1.0

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location={"cuda:0": str(device)} if device.type == "cuda" else "cpu", weights_only=False)
        if "params" in resume_ckpt:
            params = _adapt_state_dict_keys_for_model(model, resume_ckpt["params"])
            model.load_state_dict(params, strict=True)
        if "optimizer" in resume_ckpt:
            opt_state = resume_ckpt["optimizer"]
        else:
            opt_state = None
        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_psnr = float(resume_ckpt.get("best_psnr", best_psnr))
        if is_main_process():
            print(f"Resumed from: {args.resume} (epoch={start_epoch})")
    else:
        opt_state = None

    train_opt = opt["train"]
    datasets_opt = opt["datasets"]

    train_set = PairedImageFolderDataset(
        meta_dir=datasets_opt["train"]["meta_dir"],
        gt_dir=datasets_opt["train"]["gt_dir"],
        patch_size=int(train_opt.get("patch_size", 256)),
        is_train=True,
        augment=bool(train_opt.get("augment", True)),
    )

    val_set = PairedImageFolderDataset(
        meta_dir=datasets_opt["val"]["meta_dir"],
        gt_dir=datasets_opt["val"]["gt_dir"],
        patch_size=0,
        is_train=False,
        augment=False,
    )

    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_opt.get("batch_size", 4)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_opt.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(train_opt.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_opt.get("lr", 1e-4)),
        weight_decay=float(train_opt.get("weight_decay", 0.0)),
    )

    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    total_epochs = int(train_opt.get("epochs", 10))
    min_lr = float(train_opt.get("min_lr", 1e-6))

    def lr_lambda(current_epoch: int):
        if total_epochs <= 1:
            return 1.0
        t = current_epoch / float(total_epochs - 1)
        return (min_lr / float(train_opt.get("lr", 1e-4))) + (1.0 - (min_lr / float(train_opt.get("lr", 1e-4)))) * 0.5 * (
            1.0 + math.cos(math.pi * t)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    amp = bool(train_opt.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    exp_root = Path(opt.get("experiment_root", "./experiments")) / opt.get("experiment_name", "finetune")
    models_dir = exp_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    task = str(train_opt.get("task", "auto"))
    save_every = int(train_opt.get("save_every", 1))
    val_every = int(train_opt.get("val_every", 1))

    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)

        model.train()

        pbar = train_loader
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"train e{epoch+1}/{total_epochs}")

        for batch in pbar:
            meta = batch["meta"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                out = model(meta, task=task)["output"]
                out = torch.clamp(out, 0.0, 1.0)
                loss = F.l1_loss(out, gt)

            loss_mean = reduce_mean(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if is_main_process():
                pbar.set_postfix({"loss": f"{loss_mean.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step()

        if (epoch + 1) % val_every == 0:
            psnr = evaluate(model, val_loader, device=device, task=task, amp=amp)
            if is_main_process():
                print(f"Epoch {epoch+1}: val_psnr={psnr:.3f}")
            if psnr > best_psnr:
                best_psnr = psnr
                if is_main_process():
                    torch.save(
                        {
                            "params": model.state_dict(),
                            "epoch": epoch + 1,
                            "best_psnr": best_psnr,
                        },
                        models_dir / "best.pt",
                    )

        if is_main_process() and ((epoch + 1) % save_every == 0 or (epoch + 1) == total_epochs):
            torch.save(
                {
                    "params": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "optimizer": optimizer.state_dict(),
                },
                models_dir / "latest.pt",
            )
            torch.save(
                {
                    "params": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "optimizer": optimizer.state_dict(),
                },
                models_dir / f"epoch_{epoch+1:04d}.pt",
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
