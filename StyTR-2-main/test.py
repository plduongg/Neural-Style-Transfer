import argparse
import os
from pathlib import Path
from os.path import basename, splitext
from collections import OrderedDict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import models.transformer as transformer
import models.StyTR as StyTR


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def test_transform(size, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize((size, size)))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict_clean(path, map_location="cpu"):
    state_dict = safe_torch_load(path, map_location=map_location)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] if k.startswith("module.") else k
        new_state_dict[namekey] = v
    return new_state_dict


def collect_image_paths(single_path, dir_path):
    if single_path:
        return [Path(single_path)]

    if not dir_path:
        raise ValueError("You must provide either a single file path or a directory path.")

    folder = Path(dir_path)
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    paths = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if len(paths) == 0:
        raise RuntimeError(f"No valid image files found in: {dir_path}")
    return paths


def main():
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument("--content", type=str, help="Path to one content image")
    parser.add_argument("--content_dir", type=str, help="Directory of content images")
    parser.add_argument("--style", type=str, help="Path to one style image")
    parser.add_argument("--style_dir", type=str, help="Directory of style images")
    parser.add_argument("--output", type=str, default="output", help="Directory to save output images")

    parser.add_argument("--vgg", type=str, default="./experiments/vgg_normalised.pth")
    parser.add_argument("--decoder_path", type=str, default="./experiments/decoder_iter_60000.pth")
    parser.add_argument("--Trans_path", type=str, default="./experiments/transformer_iter_60000.pth")
    parser.add_argument("--embedding_path", type=str, default="./experiments/embedding_iter_60000.pth")

    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"))
    parser.add_argument("--hidden_dim", default=512, type=int)

    parser.add_argument("--content_size", type=int, default=512)
    parser.add_argument("--style_size", type=int, default=512)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--save_ext", type=str, default=".jpg")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Collect images
    content_paths = collect_image_paths(args.content, args.content_dir)
    style_paths = collect_image_paths(args.style, args.style_dir)

    os.makedirs(args.output, exist_ok=True)

    # Build model
    vgg = StyTR.vgg
    vgg.load_state_dict(safe_torch_load(args.vgg, map_location="cpu"))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.load_state_dict(load_state_dict_clean(args.decoder_path, map_location="cpu"))
    Trans.load_state_dict(load_state_dict_clean(args.Trans_path, map_location="cpu"))
    embedding.load_state_dict(load_state_dict_clean(args.embedding_path, map_location="cpu"))

    vgg.eval()
    decoder.eval()
    Trans.eval()
    embedding.eval()

    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
    network.eval()
    network.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    total = len(content_paths) * len(style_paths)
    idx = 0

    with torch.no_grad():
        for content_path in content_paths:
            for style_path in style_paths:
                idx += 1
                print(f"[{idx}/{total}] Content: {content_path.name} | Style: {style_path.name}")

                content = content_tf(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
                style = style_tf(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)

                with torch.no_grad():
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