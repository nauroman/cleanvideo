from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUPIR_ROOT = ROOT / "external" / "SUPIR"
sys.path.insert(0, str(SUPIR_ROOT))
os.chdir(SUPIR_ROOT)

from SUPIR.util import PIL2Tensor, Tensor2PIL, convert_dtype, create_SUPIR_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanVideo SUPIR folder adapter")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--SUPIR_sign", type=str, default="Q", choices=["F", "Q"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min_size", type=int, default=1024)
    parser.add_argument("--edm_steps", type=int, default=50)
    parser.add_argument("--s_stage1", type=int, default=-1)
    parser.add_argument("--s_churn", type=int, default=5)
    parser.add_argument("--s_noise", type=float, default=1.01)
    parser.add_argument("--s_cfg", type=float, default=4.0)
    parser.add_argument("--s_stage2", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--a_prompt",
        type=str,
        default=(
            "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R "
            "camera, hyper detailed photo - realistic maximum detail, 32k, Color "
            "Grading, ultra HD, extreme meticulous detailing, skin pore detailing, "
            "hyper sharpness, perfect without deformations."
        ),
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default=(
            "painting, oil painting, illustration, drawing, art, sketch, oil painting, "
            "cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, "
            "worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
            "deformed, lowres, over-smooth"
        ),
    )
    parser.add_argument("--color_fix_type", type=str, default="Wavelet", choices=["None", "AdaIn", "Wavelet"])
    parser.add_argument("--linear_CFG", action="store_true", default=True)
    parser.add_argument("--linear_s_stage2", action="store_true", default=False)
    parser.add_argument("--spt_linear_CFG", type=float, default=1.0)
    parser.add_argument("--spt_linear_s_stage2", type=float, default=0.0)
    parser.add_argument("--ae_dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--diff_dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--no_llava", action="store_true", default=False)
    parser.add_argument("--loading_half_params", action="store_true", default=False)
    parser.add_argument("--use_tile_vae", action="store_true", default=False)
    parser.add_argument("--encoder_tile_size", type=int, default=512)
    parser.add_argument("--decoder_tile_size", type=int, default=64)
    parser.add_argument("--load_8bit_llava", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args, flush=True)

    if torch.cuda.device_count() >= 1:
        supir_device = "cuda:0"
    else:
        raise ValueError("SUPIR requires CUDA.")

    model = create_SUPIR_model("options/SUPIR_v0.yaml", SUPIR_sign=args.SUPIR_sign)
    if args.loading_half_params:
        model = model.half()
    if args.use_tile_vae:
        model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
    model.ae_dtype = convert_dtype(args.ae_dtype)
    model.model.dtype = convert_dtype(args.diff_dtype)
    model = model.to(supir_device)

    llava_agent = None
    if not args.no_llava:
        from CKPT_PTH import LLAVA_MODEL_PATH  # noqa: WPS433
        from llava.llava_agent import LLavaAgent  # noqa: WPS433

        llava_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
        llava_agent = LLavaAgent(
            LLAVA_MODEL_PATH,
            device=llava_device,
            load_8bit=args.load_8bit_llava,
            load_4bit=False,
        )

    input_dir = Path(args.img_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        image_name = image_path.stem
        source = Image.open(image_path).convert("RGB")
        lq_img, h0, w0 = PIL2Tensor(source, upsacle=args.upscale, min_size=args.min_size)
        lq_img = lq_img.unsqueeze(0).to(supir_device)[:, :3, :, :]

        if llava_agent is not None:
            lq_img_512, h1, w1 = PIL2Tensor(
                source,
                upsacle=args.upscale,
                min_size=args.min_size,
                fix_resize=512,
            )
            lq_img_512 = lq_img_512.unsqueeze(0).to(supir_device)[:, :3, :, :]
            clean_imgs = model.batchify_denoise(lq_img_512)
            clean_pil_img = Tensor2PIL(clean_imgs[0], h1, w1)
            captions = llava_agent.gen_image_caption([clean_pil_img])
        else:
            captions = [""]
        print(captions, flush=True)

        samples = model.batchify_sample(
            lq_img,
            captions,
            num_steps=args.edm_steps,
            restoration_scale=args.s_stage1,
            s_churn=args.s_churn,
            s_noise=args.s_noise,
            cfg_scale=args.s_cfg,
            control_scale=args.s_stage2,
            seed=args.seed,
            num_samples=args.num_samples,
            p_p=args.a_prompt,
            n_p=args.n_prompt,
            color_fix_type=args.color_fix_type,
            use_linear_CFG=args.linear_CFG,
            use_linear_control_scale=args.linear_s_stage2,
            cfg_scale_start=args.spt_linear_CFG,
            control_scale_start=args.spt_linear_s_stage2,
        )
        for index, sample in enumerate(samples):
            Tensor2PIL(sample, h0, w0).save(save_dir / f"{image_name}_{index}.png")


if __name__ == "__main__":
    main()
