import json
import cv2
from PIL import Image
import sys
sys.path.append('../..')
from model import longclip
import torch
import torch.utils.data as data
import os
import numpy as np
from tqdm import tqdm

docci_image_root = '/SHARE_ST/icl/hyperbolic/datasets/eval/docci/pipeline/dataset/images'
docci_json_path = '/SHARE_ST/icl/hyperbolic/datasets/eval/docci/datasets/docci_test.json'


# image_root = '/SHARE_ST/icl/hyperbolic/datasets/eval/Urban1k/Urban1k/image/'
# caption_root = '/SHARE_ST/icl/hyperbolic/datasets/eval/Urban1k/Urban1k/caption/'
class local_dataset(data.Dataset):
    def __init__(self, max_samples=None):
        self.image_root = docci_image_root
        self.json_path = docci_json_path

        with open(self.json_path, 'r') as f:
            all_data = json.load(f)

        # filename에 'test'가 포함된 것만 사용
        self.data = [
            item for item in all_data
            if 'test' in os.path.basename(item['filename'])
        ]

        # 순서 고정
        self.data = sorted(self.data, key=lambda x: x['filename'])

        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # filename에서 실제 파일명만 추출
        image_name = os.path.basename(item['filename'])
        image_path = os.path.join(self.image_root, image_name)

        caption = item['caption']

        image = Image.open(image_path).convert("RGB")

        return image, caption

if __name__ == '__main__':
    dataset = local_dataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_hycoclip = True
    load_from_clip = True

    if is_hycoclip: 
        model, preprocess = longclip.load_from_clip(device='cuda',name="/home/khy5630/2025-temp/pixel/Long-CLIP/checkpoints/clip_vit_b.pth", is_hycoclip=is_hycoclip, load_from_clip=load_from_clip)
    else:
        model, preprocess = longclip.load_from_clip(device='cuda',name="/home/khy5630/2025-temp/pixel/Long-CLIP/checkpoints/ViT-B-16.pt", load_from_clip=load_from_clip)
    model.eval()
    print("model done!")
    
    img_feature_list = []
    text_list_1 = []
    text_list_2 = []
    text_list = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (image, caption) in enumerate(tqdm(dataset)):
            text_list.append(caption)

        text_feature = longclip.tokenize(text_list, truncate=True).to(device)
        text_feature = model.encode_text(text_feature)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        
        for i, (image, caption) in enumerate(tqdm(dataset)):            
            image = preprocess(image).unsqueeze(0).to(device)
            img_feature = model.encode_image(image)
            img_feature_list.append(img_feature)
            
        image_embeds = torch.cat(img_feature_list, dim=0)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
        print("text 2 image")
        i = 0
        correct = 0
        total = 0
        for i in range(text_feature.shape[0]):
            text = text_feature[i]
            sim = text @ image_embeds.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)
        
        print("image to text")
        i = 0
        correct = 0
        total = 0
        for i in range(image_embeds.shape[0]):
            img = image_embeds[i]
            sim = img @ text_feature.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)

