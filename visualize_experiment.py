import csv
import argparse
from pathlib import Path

import torch


def plot_results_png(csv_path: str | Path, out_png: str | Path):
    csv_path = Path(csv_path)
    out_png = Path(out_png)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not csv_path.is_file():
        return

    epochs: list[int] = []
    train_psnr: list[float] = []
    train_ssim: list[float] = []
    val_psnr: list[float] = []
    val_ssim: list[float] = []

    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                epochs.append(int(row["epoch"]))
                train_psnr.append(float(row["train_psnr"]))
                train_ssim.append(float(row["train_ssim"]))
                val_psnr.append(float(row["val_psnr"]))
                val_ssim.append(float(row["val_ssim"]))
            except Exception:
                continue

    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=120, sharex=True)
    ax0, ax1 = axes

    ax0.plot(epochs, train_psnr, label="train_psnr")
    ax0.plot(epochs, val_psnr, label="val_psnr")
    ax0.set_ylabel("PSNR")
    ax0.grid(True, linestyle="--", alpha=0.3)
    ax0.legend()

    ax1.plot(epochs, train_ssim, label="train_ssim")
    ax1.plot(epochs, val_ssim, label="val_ssim")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("SSIM")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


@torch.no_grad()
def save_vis_grid(
    model,
    loader,
    device: torch.device,
    out_path: str | Path,
    task: str = "auto",
    amp: bool = True,
    max_samples: int = 4,
):
    out_path = Path(out_path)

    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    meta = batch["meta"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    meta = meta[:max_samples]
    gt = gt[:max_samples]

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            pred = model(meta, task=task)["output"]
    finally:
        if was_training:
            model.train()

    meta = torch.clamp(meta, 0.0, 1.0)
    pred = torch.clamp(pred, 0.0, 1.0)
    gt = torch.clamp(gt, 0.0, 1.0)

    triplets = torch.cat([meta, pred, gt], dim=3)

    try:
        from torchvision.utils import make_grid, save_image
    except Exception:
        return

    grid = make_grid(triplets, nrow=1, padding=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize DeMoE experiments")
    parser.add_argument("--csv", type=str, default="", help="Path to result.csv")
    parser.add_argument("--exp_root", type=str, default="", help="Path to experiments/<run> directory")
    parser.add_argument("--out", type=str, default="", help="Output results.png path (optional)")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    exp_root = Path(args.exp_root) if args.exp_root else None

    if csv_path is None:
        if exp_root is None:
            raise SystemExit("Please provide --csv or --exp_root")
        csv_path = exp_root / "result.csv"

    if exp_root is None:
        exp_root = csv_path.parent

    out_png = Path(args.out) if args.out else (exp_root / "results.png")
    plot_results_png(csv_path, out_png)


if __name__ == "__main__":
    main()
