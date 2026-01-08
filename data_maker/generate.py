import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from nano import fft, ifft, psf2otf
from nano import ensure_psf_npy

def pick_device():
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return "mps"
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
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--poss", type=float, default=4e-5)
    parser.add_argument("--gaus", type=float, default=1e-5)
    args = parser.parse_args()

    device = pick_device()

    input_path = Path(args.input)
    psf_path = Path(args.psf)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    ensure_psf_npy(psf_path)

    psf = np.load(str(psf_path))
    center_idx = psf.shape[0] // 2
    psf_center = torch.from_numpy(psf[center_idx]).to(device)
    pad = psf.shape[1]

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if input_path.is_dir():
        input_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts])
    else:
        input_files = [input_path]

    for fp in input_files:
        img = Image.open(fp).convert("RGB")
        img_np = np.asarray(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        try:
            out = apply_convolution_tiled(
                img_t,
                psf_center,
                pad=pad,
                tile=int(args.tile),
                poss=float(args.poss),
                gaus=float(args.gaus),
            )
        except Exception:
            if device == "mps":
                device = "cpu"
                img_t = img_t.to(device)
                psf_center = psf_center.to(device)
                out = apply_convolution_tiled(
                    img_t,
                    psf_center,
                    pad=pad,
                    tile=int(args.tile),
                    poss=float(args.poss),
                    gaus=float(args.gaus),
                )
            else:
                raise

        out_np = out.permute(1, 2, 0).numpy()
        out_img = Image.fromarray((out_np * 255.0).round().astype(np.uint8))
        out_file = output_path / f"{fp.stem}_lensless_centerpsf.png"
        out_img.save(out_file)
        print("saved:", str(out_file))

    print("device:", device)


def _add_noise(image, poss=4e-5, gaus=1e-5):
    try:
        poss_img = torch.poisson(torch.clamp(image / poss, min=0.0)) * poss
    except Exception:
        poss_img = torch.poisson(torch.clamp((image / poss).cpu(), min=0.0)).to(image.device) * poss
    gauss_noise = torch.randn_like(image) * gaus
    return torch.clamp(poss_img + gauss_noise, 0.0, 1.0)


def apply_convolution_tiled(image, psf, pad, tile=512, poss=4e-5, gaus=1e-5):
    b, c, h, w = image.shape
    block = tile + 2 * pad

    n_tiles_y = (h + tile - 1) // tile
    n_tiles_x = (w + tile - 1) // tile
    total_h = n_tiles_y * tile + 2 * pad
    total_w = n_tiles_x * tile + 2 * pad

    pad_right = total_w - w - pad
    pad_bottom = total_h - h - pad

    image_pad = F.pad(image, (pad, pad_right, pad, pad_bottom))
    otf = psf2otf(psf, h=block, w=block, permute=True).to(image.device)

    out_cpu = torch.empty((c, h, w), dtype=torch.float32)

    with torch.no_grad():
        for ty in range(n_tiles_y):
            y0 = ty * tile
            for tx in range(n_tiles_x):
                x0 = tx * tile
                block_img = image_pad[..., y0:y0 + block, x0:x0 + block]
                conv = ifft(fft(block_img) * otf)
                conv = torch.clamp(conv, min=1e-20, max=1.0)
                conv = _add_noise(conv, poss=poss, gaus=gaus)

                tile_out = conv[..., pad:pad + tile, pad:pad + tile]
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)
                out_cpu[:, y0:y1, x0:x1] = tile_out[0, :, :y1 - y0, :x1 - x0].cpu()

    return out_cpu


if __name__ == "__main__":
    main()