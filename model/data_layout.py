import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    box_images = torch.stack([b["box_image"] for b in batch])

    texts = [b["text"] for b in batch]

    box_texts = []
    for b in batch:
        box_texts.extend(b["box_text"])

    B, NB = box_images.shape[:2]
    box_images = box_images.view(B * NB, *box_images.shape[2:])

    return images, box_images, texts, box_texts



class LayoutSAMJsonDataset(Dataset):
    def __init__(
        self,
        json_path: str,          # 🔥 폴더
        image_root: str,
        image_size: int = 224,
        # max_boxes: int = 5,
        max_boxes: int = 1,
        min_score: float = 0.0,
    ):
        """
        Args:
            json_dir: directory containing per-image json files
            image_root: root directory for images
        """
        self.json_path = json_path
        self.image_root = image_root
        self.max_boxes = max_boxes
        self.min_score = min_score

        # 🔥 모든 json 파일 수집
        self.json_files = sorted([
            os.path.join(json_path, f)
            for f in os.listdir(json_path)
            if f.endswith(".json")
        ])

        if len(self.json_files) == 0:
            raise RuntimeError(f"No json files found in {json_path}")

        self.transforms = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

        self.tensor_resize = T.Resize((image_size, image_size))
        # self.transforms = [TransformWithBoxes(t) for t in base_transforms]
        # self.data = []
        # for json_file in self.json_files:
        #     with open(json_file, "r") as f:
        #         self.data.append(json.load(f))


    def apply_transform_safe(self, x):
        # PIL Image
        if isinstance(x, Image.Image):
            return self.transforms(x)

        # Tensor (C,H,W)
        elif torch.is_tensor(x):
            return self.tensor_resize(x)

        else:
            raise TypeError(f"Unsupported type for transform: {type(x)}")

    def apply_transforms(self, image, boxes):
        transformed_image = self.apply_transform_safe(image)
        transformed_boxes = []
        for box in boxes:
            transformed_boxes.append(self.apply_transform_safe(image[box]))
        transformed_boxes  = torch.cat(transformed_boxes, dim=0)
        return transformed_image, transformed_boxes

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # sample = self.data[idx]
        with open(self.json_files[idx], "r") as f:
            sample = json.load(f)

        # ---- load image ----
        filename = os.path.basename(sample["image_path"])
        img_path = os.path.join(self.image_root, filename)

        # try: 
        #     image = Image.open(img_path).convert("RGB")


        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load image: {img_path}")
            new_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_idx)
            
        W, H = image.size

        # ---- parse bbox info ----
        bbox_infos = sample["metadata"]["bbox_info"]

        boxes, texts = [], []
        for obj in bbox_infos:
            if obj["score"] < self.min_score:
                continue
            boxes.append(obj["bbox"])
            texts.append(obj["description"])

        if len(boxes) == 0:
            boxes = [[0, 0, W, H]]
            texts = ["background"]

        boxes = torch.tensor(boxes, dtype=torch.float32)

        # # ---- pixel → normalized ----
        # boxes[:, [0, 2]] /= W
        # boxes[:, [1, 3]] /= H

        num_boxes = boxes.shape[0]
        limit = min(num_boxes, self.max_boxes)

        

        # ---- apply joint transforms ----
        # image, boxes = self.apply_transforms(image, boxes)
        image = self.apply_transform_safe(image)
        

        # ---- box images (crop from transformed image) ----
        box_images = torch.zeros((self.max_boxes, 3, 224, 224))
        box_texts = []

        _, h, w = image.shape
        for i in range(limit):
            x1, y1, x2, y2 = boxes[i]
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)

            crop = image[:, py1:py2, px1:px2]
            if crop.numel() > 0:
                # crop = T.Resize((224,224))(crop)
                box_images[i] = self.apply_transform_safe(crop)

            box_texts.append(texts[i])

        # ---- bbox template ----
        bbox_template = torch.zeros((self.max_boxes, 5))
        bbox_template[:limit, :4] = boxes[:limit]
        bbox_template[:limit, 4] = 1.0

        # ---- global text (optional) ----
        global_text = ", ".join(box_texts)

        return {
            "image": image,
            "text": global_text,
            "box_image": box_images,
            "box_text": box_texts,
            "bbox": bbox_template,
            "num_boxes": limit,
        }

