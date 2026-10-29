import argparse
import os
from pathlib import Path
from os.path import basename, splitext
from collections import OrderedDict

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.utils import save_image

import models.transformer as transformer
import models.StyTR as StyTR

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_torch_load(path, map_location="cpu", weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict_clean(path, map_location="cpu"):
    state_dict = safe_torch_load(path, map_location=map_location, weights_only=False)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] if k.startswith("module.") else k
        new_state_dict[namekey] = v
    return new_state_dict


def collect_image_paths(single_path, dir_path):
    if single_path:
        p = Path(single_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {single_path}")
        return [p]

    if not dir_path:
        raise ValueError("You must provide either --content/--style or --content_dir/--style_dir")

    folder = Path(dir_path)
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    paths = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if len(paths) == 0:
        raise RuntimeError(f"No valid image files found in: {dir_path}")
    return paths


def test_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content", type=str, default=None)
    parser.add_argument("--content_dir", type=str, default=None)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--style_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/rpa_60k")

    parser.add_argument("--vgg", type=str, default="./experiments/vgg_normalised.pth")
    parser.add_argument("--decoder_path", type=str, required=True)
    parser.add_argument("--Trans_path", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)

    parser.add_argument("--style_interpolation_weights", type=str, default="")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"))
    parser.add_argument("--hidden_dim", default=512, type=int)

    # Khớp với file transformer RPA
    parser.add_argument("--use_rel_pos_attention", dest="use_rel_pos_attention", action="store_true")
    parser.add_argument("--no_rel_pos_attention", dest="use_rel_pos_attention", action="store_false")
    parser.set_defaults(use_rel_pos_attention=True)
    parser.add_argument("--max_rel_pos", type=int, default=64)

    parser.add_argument("--content_size", type=int, default=192)
    parser.add_argument("--style_size", type=int, default=192)
    parser.add_argument("--save_ext", type=str, default=".jpg")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"use_rel_pos_attention={args.use_rel_pos_attention}, max_rel_pos={args.max_rel_pos}")

    content_paths = collect_image_paths(args.content, args.content_dir)
    style_paths = collect_image_paths(args.style, args.style_dir)

    os.makedirs(args.output, exist_ok=True)

    vgg = StyTR.vgg
    vgg.load_state_dict(safe_torch_load(args.vgg, map_location="cpu", weights_only=True))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    embedding = StyTR.PatchEmbed()

    Trans = transformer.Transformer(
        d_model=args.hidden_dim,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        use_rel_pos=args.use_rel_pos_attention,
        max_pos_size=args.max_rel_pos,
    )

    decoder.load_state_dict(load_state_dict_clean(args.decoder_path, map_location="cpu"))
    Trans.load_state_dict(load_state_dict_clean(args.Trans_path, map_location="cpu"))
    embedding.load_state_dict(load_state_dict_clean(args.embedding_path, map_location="cpu"))

    vgg.eval()
    decoder.eval()
    embedding.eval()
    Trans.eval()

    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
    network.eval().to(device)

    content_tf = test_transform(args.content_size)
    style_tf = test_transform(args.style_size)

    total = len(content_paths) * len(style_paths)
    idx = 0

    with torch.no_grad():
        for content_path in content_paths:
            for style_path in style_paths:
                idx += 1
                print(f"[{idx}/{total}] Content: {content_path.name} | Style: {style_path.name}")

                content = content_tf(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
                style = style_tf(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)

                output = network(content, style)

                if isinstance(output, tuple):
                    output = output[0]

                output = output.detach().cpu()

                output_name = os.path.join(
                    args.output,
                    f"{splitext(basename(content_path))[0]}_stylized_{splitext(basename(style_path))[0]}{args.save_ext}"
                )
                save_image(output, output_name)

    print(f"Done. Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()