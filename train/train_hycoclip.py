# ============================================================
# Imports
# ============================================================
import os
import json
import argparse
import subprocess
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


import sys
sys.path.append("..")

from scheduler import cosine_lr
from model import longclip
import model.lorentz as L   # <-- lorentz.py 그대로 사용
from model.data_layout import LayoutSAMJsonDataset, collate_fn

# ============================================================
# Distributed setup
# ============================================================
def setup_distributed(backend="nccl"):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        os.environ.setdefault("MASTER_PORT", "29522")
        os.environ.setdefault("MASTER_ADDR", addr)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, rank % num_gpus
# ============================================================
# Trainer
# ============================================================
class Trainer:
    def __init__(self, rank, local_rank, args):
        self.rank = rank

        is_hycoclip = True
        load_from_clip = True
        # model, preprocess = longclip.load_from_clip(device='cuda',name="/home/khy5630/2025-temp/pixel/Long-CLIP/checkpoints/clip_vit_b.pth", is_hycoclip=is_hycoclip, load_from_clip=load_from_clip)
        model, preprocess = longclip.load_from_clip(device='cuda',name="/home/khy5630/2025-temp/pixel/Long-CLIP/checkpoints/hycoclip_vit_b.pth", is_hycoclip=is_hycoclip, load_from_clip=load_from_clip)
        self.model = model.cuda()

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scaler = GradScaler()
        self.epochs = args.epochs
        self.global_step = 0

    def train(self, loader, scheduler):
        for epoch in range(self.epochs):
            loader.sampler.set_epoch(epoch)
            self.model.train()
            print(f"--------------epoch: {epoch}--------------")

            for step, (images, box_images, texts, box_texts) in tqdm(enumerate(loader)):
                images = images.cuda(non_blocking=True)
                box_images = box_images.cuda(non_blocking=True)

                tokens = longclip.tokenize(texts, truncate=True).cuda()
                box_tokens = longclip.tokenize(box_texts, truncate=True).cuda()

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(dtype=torch.float32):
                    out = self.model.module.forward_hycoclip(
                        images=images,
                        box_images=box_images,
                        tokens=tokens,
                        box_tokens=box_tokens,
                        rank=self.rank
                    )

                # ❗ loss는 forward에서 정의된 그대로 사용
                self.scaler.scale(out["loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # scheduler.step()
                scheduler(self.global_step) 
                self.global_step += 1

                if self.rank == 0 and step % 50 == 0:
                    print(
                        f"[E{epoch} S{step}] "
                        f"loss={out['loss'].item():.4f} "
                        f"contrast={out['contrastive_loss'].item():.4f}"
                    )

            if self.rank == 0:
                torch.save(
                    self.model.module.state_dict(),
                    f"ckpt_epoch_{epoch}.pt",
                )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_root", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_length", type=int, default=200)
    args = parser.parse_args()

    rank, local_rank = setup_distributed()

    dataset = LayoutSAMJsonDataset(
        json_path=args.json_root,
        image_root=args.image_root,
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=32,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    trainer = Trainer(rank, local_rank, args)

    scheduler = cosine_lr(
        trainer.optimizer,
        base_lr=args.lr,
        warmup_length=args.warmup_length,
        steps=args.epochs * len(loader),
    )

    trainer.train(loader, scheduler)
