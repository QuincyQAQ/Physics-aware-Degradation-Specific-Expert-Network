import numpy as np
import torch
from PIL import Image
from pathlib import Path

from nano import apply_convolution
from nano import ensure_psf_npy


def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    root = Path(__file__).resolve().parent
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--input", type=str, default=str(root / "raw_data"))
    parser.add_argument("--psf", type=str, default=str(root / "psf.npy"))
    parser.add_argument("--output", type=str, default=str(root / "result_data"))
    parser.add_argument("--max_side", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()

    input_path = Path(args.input)
    psf_path = Path(args.psf)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    ensure_psf_npy(psf_path)

    psf = np.load(str(psf_path))  # (81,135,135,3)
    center_idx = psf.shape[0] // 2  # 40，对应 9x9 的中心 PSF
    psf_center = torch.from_numpy(psf[center_idx]).to(device)  # (H,W,C)
    pad = psf.shape[1]  # 135

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if input_path.is_dir():
        input_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts])
    else:
        input_files = [input_path]

    for fp in input_files:
        img = Image.open(fp).convert("RGB")
        W, H = img.size

        if args.max_side and args.max_side > 0:
            max_side = int(args.max_side)
            scale = max_side / max(W, H)
            if scale < 1:
                img = img.resize((int(round(W * scale)), int(round(H * scale))), Image.BICUBIC)

        img_np = np.asarray(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                out, _ = apply_convolution(img_t, psf_center, pad=pad)
        except Exception:
            if device == "mps":
                device = "cpu"
                img_t = img_t.to(device)
                psf_center = psf_center.to(device)
                with torch.no_grad():
                    out, _ = apply_convolution(img_t, psf_center, pad=pad)
            else:
                raise

        out = out[..., pad:-pad, pad:-pad].clamp(0, 1)
        out_np = out[0].permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray((out_np * 255.0).round().astype(np.uint8))
        out_file = output_path / f"{fp.stem}_lensless_centerpsf_resize.png"
        out_img.save(out_file)
        print("saved:", str(out_file))

    print("device:", device)


if __name__ == "__main__":
    main()