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
    model = ERGP(in_ch=3, n_class=2, bilinear=True,).to(args.device)
    state_dict = torch.load(args.pre_trained, map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_IOU(pred,mask):
    metric = SegmentationMetric(2)
    metric.addBatch(pred, mask)
    pa = metric.pixelAccuracy()
    # cpa = metric.classPixelAccuracy()
    # Recall = metric.classPixelRecall()
    mpa = metric.meanPixelAccuracy()
    mre = metric.meanPixelRecall()
    mIoU = metric.meanIntersectionOverUnion()
    dice = 2*mpa*mre/(mpa+mre)

    return pa,  mpa, mre, mIoU, dice


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Arguments

    parser.add_argument('--data-folder', type=str, default='../Dataset/COVID/Val(COVID-19)/images', )
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 8)')
    parser.add_argument('--pre-trained', type=str, default="./scratch/4_0.0001_100/models/best.pt",
                        help='path of pre-trained weights (default: None)')

    args = parser.parse_args()
    args.device = torch.device("cuda")

    image_files = sorted(glob('{}/*.pn*g'.format(args.data_folder)))
    model = load_model()
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')
    print(len(image_files), ' files in folder ', args.data_folder)

    fig = plt.figure()
    f = open("./pre_txt/pred_.txt", "w")
    iou = []
    PA = []
    MPA = []
    MRE = []
    Mdice = []
    for i, image_file in enumerate(image_files):
        # if i >= args.batch_size:
        # break
        mask_file = re.sub("images", "infection_masks", image_file)
        mask_file = re.sub("png", "png", mask_file)
        image = cv2.imread(image_file)
        image_name = image_file.split("\\")[-1]
        mask_name = re.sub("png","png",image_name)
        mask1 = cv2.imread(mask_file)
        # mask1 = cv2.resize(mask1,(224,224))
        mask1 = np.transpose(mask1,(2,0,1))
        mask1 = mask1[0]
        mask1[mask1>0.1] = 1
        mask2 = mask1 * 255
        img_size = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        pil_img = Image.fromarray(image)
        torch_img = transform(pil_img)
        torch_img = torch_img.unsqueeze(0)
        torch_img = torch_img.to(args.device)

        # Forward Pass
        logits = model(torch_img)
        pred = np.argmax(logits.data.cpu().numpy(), axis=1)
        pred = pred.squeeze()
        pa, mpa, mre, mIoU, dice = get_IOU(pred, mask1)
        mask = np.argmax(logits.data.cpu().numpy(), axis=1)
        mask_s = np.array(mask, dtype=np.uint8) * 255
        mask_s = mask_s.squeeze()

        # mask90 = np.rot90(masks)
        print("pa: {}  mpa: {}  mre:{}  miou: {}  dice:{}".format(pa,mpa,mre,mIoU,dice))
        # cv2.imshow("mask-T", cv2.resize(mask2,(600,600)))
        # cv2.imshow("mask-P", cv2.resize(mask_s,(600,600)))
        cv2.imwrite('./pre_466/{}'.format(mask_name), mask_s)
        cv2.waitKey(2000)
        f.write("pa: {:.4f}   mpa: {:.4f}  mre{:.4f}  miou: {:.4f}  dice:{:.4f}".format(pa,mpa,mre,mIoU,dice))
        # f.write("miou: {:.6f}".format(mIoU))

        PA.append(pa)
        MPA.append(mpa)
        MRE.append(mre)
        iou.append(mIoU)
        Mdice.append(dice)

        f.write("\n")
        f.flush()
    f.write("\nmean pa:  {}".format(np.mean(PA)))
    f.write("\nmean mpa:  {}".format(np.mean(MPA)))
    f.write("\nmean mre:  {}".format(np.mean(MRE)))
    f.write("\nmean iou:  {}".format(np.mean(iou)))
    f.write("\nmean dice:  {}".format(np.mean(Mdice)))
    f.close()



# # Plot
#     ax = plt.subplot(2, args.batch_size, 2 * i + 1)
#     ax.axis('off')
#     ax.imshow(image.squeeze())
#
#     ax = plt.subplot(2, args.batch_size, 2 * i + 2)
#     ax.axis('off')
#     ax.imshow(mask.squeeze())
#
#     plt.show()
