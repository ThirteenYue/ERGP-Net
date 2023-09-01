from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from src import ERGP
import os
from MLtrainer import SegmentationMetric
import re

# load pre-trained model and weights
def load_model():
    model = ERGP(in_ch=3, n_class=2, bilinear=True, ).to(args.device)
    state_dict = torch.load(args.pre_trained, map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()
    return model

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Arguments
    img_path = "../Dataset/COVID/Val(COVID-19)/images/covid_1919.png"
    roi_mask_path = "../Dataset/COVID/Val(COVID-19)/lung_masks/covid_1919.png"

    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--pre-trained', type=str, default="./scratch/18324/models/best.pt",
                        help='path of pre-trained weights (default: None)')

    args = parser.parse_args()
    args.device = torch.device("cuda")

    model = load_model()
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)
    # roi_img = transform(roi_img)
    # roi_img = torch.unsqueeze(roi_img*255, dim=0)

    original_img = Image.open(img_path).convert('RGB')
    img = transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    # roi_img = transform(img)
    # img = np.transpose(img, (0, 3, 1, 2))
    # img = img.to(args.device)

    output = model(img.to(args.device))

    prediction = output[0].argmax(0).squeeze(0)
    # roi_img = roi_img[0].argmax(0).squeeze(0)

    prediction = prediction.to("cpu").numpy().astype(np.uint8)
    # 将前景对应的像素值改成255(白色)
    prediction[prediction == 1] = 255
    # 将不敢兴趣的区域像素设置成0(黑色)
    prediction[roi_img == 0] = 100
    mask = Image.fromarray(prediction).convert('RGB')
    mask = Image.blend(original_img,mask,0.7)

    mask.save("./pre/test_result2.png")




