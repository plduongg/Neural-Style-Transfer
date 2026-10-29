import argparse
import os
import re
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import models.transformer as transformer
import models.StyTR as StyTR
from sampler import InfiniteSamplerWrapper


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(192),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)

        self.path = os.listdir(self.root)
        if len(self.path) == 0:
            raise RuntimeError(f"Empty folder: {self.root}")

        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                folder_path = os.path.join(self.root, file_name)
                if not os.path.isdir(folder_path):
                    continue
                for file_name1 in os.listdir(folder_path):
                    self.paths.append(os.path.join(folder_path, file_name1))
        else:
            self.paths = list(Path(self.root).glob('*'))

        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def safe_torch_load(path, map_location='cpu', weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    else:
        return obj


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def find_latest_iteration(save_dir):
    pattern = re.compile(r'^transformer_iter_(\d+)\.pth$')
    iterations = []

    if not os.path.exists(save_dir):
        return 0

    for file_name in os.listdir(save_dir):
        match = pattern.match(file_name)
        if match:
            iteration = int(match.group(1))
            decoder_path = os.path.join(save_dir, f'decoder_iter_{iteration}.pth')
            embedding_path = os.path.join(save_dir, f'embedding_iter_{iteration}.pth')
            if os.path.exists(decoder_path) and os.path.exists(embedding_path):
                iterations.append(iteration)

    return max(iterations) if iterations else 0


parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content_dir', default='./data/train', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./data/style', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=60000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=5000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help='Type of positional embedding to use on top of the image features')
parser.add_argument('--hidden_dim', default=512, type=int,
                    help='Size of the embeddings (dimension of the transformer)')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='CUDA device id')

# resume options
parser.add_argument('--resume_iter', type=int, default=0,
                    help='Resume from iteration, e.g. 15000')
parser.add_argument('--auto_resume', action='store_true',
                    help='Automatically resume from latest checkpoint in save_dir if found')

args = parser.parse_args()

if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()

    if USE_CUDA:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        try:
            torch.cuda.set_per_process_memory_fraction(0.95, device=args.gpu_id)
        except TypeError:
            torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device("cpu")
        print("[DEVICE] CUDA not available, fallback to CPU")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "test"), exist_ok=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    vgg = StyTR.vgg
    vgg.load_state_dict(safe_torch_load(args.vgg, map_location='cpu', weights_only=True))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    embedding = StyTR.PatchEmbed()
    Trans = transformer.Transformer()

    with torch.no_grad():
        network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)

    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads,
        pin_memory=USE_CUDA
    ))

    style_iter = iter(data.DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads,
        pin_memory=USE_CUDA
    ))

    optimizer = torch.optim.Adam([
        {'params': network.transformer.parameters()},
        {'params': network.decode.parameters()},
        {'params': network.embedding.parameters()},
    ], lr=args.lr)

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=USE_CUDA)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)

    # -----------------------------
    # Resume logic
    # -----------------------------
    start_iter = 0
    resume_iter = args.resume_iter

    if args.auto_resume and resume_iter == 0:
        resume_iter = find_latest_iteration(args.save_dir)
        if resume_iter > 0:
            print(f'[AUTO RESUME] Found latest iteration: {resume_iter}')

    if resume_iter > 0:
        full_ckpt_path = os.path.join(args.save_dir, f'checkpoint_iter_{resume_iter}.pth')
        transformer_path = os.path.join(args.save_dir, f'transformer_iter_{resume_iter}.pth')
        decoder_path = os.path.join(args.save_dir, f'decoder_iter_{resume_iter}.pth')
        embedding_path = os.path.join(args.save_dir, f'embedding_iter_{resume_iter}.pth')

        if os.path.exists(full_ckpt_path):
            print(f'[RESUME] Loading full checkpoint: {full_ckpt_path}')
            checkpoint = safe_torch_load(full_ckpt_path, map_location='cpu', weights_only=False)

            network.transformer.load_state_dict(checkpoint['transformer'])
            network.decode.load_state_dict(checkpoint['decoder'])
            network.embedding.load_state_dict(checkpoint['embedding'])

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                move_optimizer_to_device(optimizer, device)

            if 'scaler' in checkpoint and USE_CUDA:
                try:
                    scaler.load_state_dict(checkpoint['scaler'])
                except Exception:
                    print('[RESUME] Could not restore scaler state, continuing with fresh scaler.')

            start_iter = checkpoint.get('iter', resume_iter)
            print(f'[RESUME] Resumed full training state from iteration {start_iter}')

        else:
            if not os.path.exists(transformer_path):
                raise FileNotFoundError(f'Not found: {transformer_path}')
            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f'Not found: {decoder_path}')
            if not os.path.exists(embedding_path):
                raise FileNotFoundError(f'Not found: {embedding_path}')

            print(f'[RESUME] Loading model weights only from iteration {resume_iter}')
            network.transformer.load_state_dict(safe_torch_load(transformer_path, map_location='cpu'))
            network.decode.load_state_dict(safe_torch_load(decoder_path, map_location='cpu'))
            network.embedding.load_state_dict(safe_torch_load(embedding_path, map_location='cpu'))

            start_iter = resume_iter
            print(f'[RESUME] Resumed model weights from iteration {start_iter}')

    if start_iter >= args.max_iter:
        raise ValueError(
            f'resume_iter/start_iter ({start_iter}) must be smaller than max_iter ({args.max_iter}).'
        )

    # -----------------------------
    # Train loop
    # -----------------------------
    for i in tqdm(range(start_iter, args.max_iter)):

        if i < 1e4:
            warmup_learning_rate(optimizer, iteration_count=i)
        else:
            adjust_learning_rate(optimizer, iteration_count=i)

        content_images = next(content_iter).to(device, non_blocking=USE_CUDA)
        style_images = next(style_iter).to(device, non_blocking=USE_CUDA)

        if USE_CUDA:
            try:
                autocast_context = torch.amp.autocast(device_type='cuda', enabled=True)
            except TypeError:
                autocast_context = torch.cuda.amp.autocast(enabled=True)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            out, loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)
            loss_c = args.content_weight * loss_c
            loss_s = args.style_weight * loss_s
            loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)

        if i % 100 == 0:
            output_name = '{:s}/test/{:s}{:s}'.format(
                args.save_dir, str(i), ".jpg"
            )
            out_vis = torch.cat((content_images, out.float()), 0)
            out_vis = torch.cat((style_images, out_vis), 0)
            save_image(out_vis, output_name)

        print(
            loss.sum().detach().cpu().numpy(),
            "-content:", loss_c.sum().detach().cpu().numpy(),
            "-style:", loss_s.sum().detach().cpu().numpy(),
            "-l1:", l_identity1.sum().detach().cpu().numpy(),
            "-l2:", l_identity2.sum().detach().cpu().numpy()
        )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss.sum()).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
        writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
        writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
        writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
        writer.add_scalar('total_loss', loss.sum().item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            transformer_state = move_to_cpu(network.transformer.state_dict())
            decoder_state = move_to_cpu(network.decode.state_dict())
            embedding_state = move_to_cpu(network.embedding.state_dict())

            torch.save(
                transformer_state,
                '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir, i + 1)
            )
            torch.save(
                decoder_state,
                '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir, i + 1)
            )
            torch.save(
                embedding_state,
                '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir, i + 1)
            )

            full_ckpt = {
                'iter': i + 1,
                'transformer': transformer_state,
                'decoder': decoder_state,
                'embedding': embedding_state,
                'optimizer': move_to_cpu(optimizer.state_dict()),
                'scaler': scaler.state_dict(),
            }

            torch.save(
                full_ckpt,
                '{:s}/checkpoint_iter_{:d}.pth'.format(args.save_dir, i + 1)
            )

            print(f'[SAVE] Saved checkpoint at iteration {i + 1}')

    writer.close()