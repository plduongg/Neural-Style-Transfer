import argparse
from pathlib import Path
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torchvision import transforms

import models.StyTR as StyTR
from function import calc_mean_std, normal

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_torch_load(path, map_location="cpu", weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_vgg_encoder(vgg_path, device):
    vgg = StyTR.vgg
    vgg.load_state_dict(safe_torch_load(vgg_path, map_location="cpu", weights_only=True))
    vgg = nn.Sequential(*list(vgg.children())[:44]).to(device).eval()

    layers = list(vgg.children())
    enc_1 = nn.Sequential(*layers[:4]).to(device).eval()
    enc_2 = nn.Sequential(*layers[4:11]).to(device).eval()
    enc_3 = nn.Sequential(*layers[11:18]).to(device).eval()
    enc_4 = nn.Sequential(*layers[18:31]).to(device).eval()
    enc_5 = nn.Sequential(*layers[31:44]).to(device).eval()

    for enc in [enc_1, enc_2, enc_3, enc_4, enc_5]:
        for p in enc.parameters():
            p.requires_grad = False

    return [enc_1, enc_2, enc_3, enc_4, enc_5]


def encode_with_intermediate(x, encoders):
    feats = []
    h = x
    for enc in encoders:
        h = enc(h)
        feats.append(h)
    return feats


def load_img(path, size, device):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def calc_content_diff(out_feats, content_feats):
    mse = nn.MSELoss()
    return (
        mse(normal(out_feats[-1]), normal(content_feats[-1])) +
        mse(normal(out_feats[-2]), normal(content_feats[-2]))
    ).item()


def calc_style_loss_single(feat_a, feat_b):
    mse = nn.MSELoss()
    a_mean, a_std = calc_mean_std(feat_a)
    b_mean, b_std = calc_mean_std(feat_b)
    return mse(a_mean, b_mean) + mse(a_std, b_std)


def calc_style_diff(out_feats, style_feats):
    loss = 0.0
    for fo, fs in zip(out_feats, style_feats):
        loss += calc_style_loss_single(fo, fs).item()
    return loss


def parse_output_name(output_name):
    stem = Path(output_name).stem
    if "_stylized_" not in stem:
        raise ValueError(f"Unexpected output filename format: {output_name}")
    return stem.split("_stylized_", 1)


def find_image_by_stem(folder, stem):
    folder = Path(folder)
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find image '{stem}' in {folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vgg", type=str, default="./experiments/vgg_normalised.pth")
    parser.add_argument("--image_size", type=int, default=192)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    output_dir = Path(args.output_dir)

    encoders = build_vgg_encoder(args.vgg, device)

    output_files = sorted([p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if len(output_files) == 0:
        raise RuntimeError(f"No output images found in {output_dir}")

    content_scores = []
    style_scores = []

    for out_path in output_files:
        content_stem, style_stem = parse_output_name(out_path.name)

        content_path = find_image_by_stem(content_dir, content_stem)
        style_path = find_image_by_stem(style_dir, style_stem)

        out_img = load_img(out_path, args.image_size, device)
        content_img = load_img(content_path, args.image_size, device)
        style_img = load_img(style_path, args.image_size, device)

        out_feats = encode_with_intermediate(out_img, encoders)
        content_feats = encode_with_intermediate(content_img, encoders)
        style_feats = encode_with_intermediate(style_img, encoders)

        content_scores.append(calc_content_diff(out_feats, content_feats))
        style_scores.append(calc_style_diff(out_feats, style_feats))

    avg_content = sum(content_scores) / len(content_scores)
    avg_style = sum(style_scores) / len(style_scores)

    print("=" * 60)
    print(f"Num outputs: {len(output_files)}")
    print(f"Average Content Difference: {avg_content:.6f}")
    print(f"Average Style Difference:   {avg_style:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()