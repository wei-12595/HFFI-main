import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
import imageio
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from lib.HPE import HPEModel
from utils.dataloader import test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=512, help='testing size')
    parser.add_argument('--data_root', type=str, default='/home/weimengru/dataset/CrackMap/data',
                        help='dataset root')
    parser.add_argument('--pth_path', type=str, required=True, help='path to best_epoch.pth')
    parser.add_argument('--save_root', type=str, default='/home/weimengru/dataset/runs/baseline',
                        help='root dir to save predictions')

    # 用于区分保存的 PR/ROC 文件后缀（例如：_clean, _gaussian）
    parser.add_argument('--tag', type=str, default='clean', help='tag for saving csv files')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 1. 路径配置
    images_dir = os.path.join(opt.data_root, 'images')
    labels_dir = os.path.join(opt.data_root, 'labels')
    test_txt = os.path.join(opt.data_root, 'annotations', 'test.txt')
    save_path = os.path.join(opt.save_root, 'predictions')
    os.makedirs(save_path, exist_ok=True)

    # 2. 模型加载
    print("Loading model from:", opt.pth_path)
    model = HPEModel()
    model.load_state_dict(torch.load(opt.pth_path, map_location=device))
    model.to(device)
    model.eval()


    # 3. 数据集加载
    test_loader = test_dataset(images_dir, labels_dir, test_txt, opt.testsize)
    print(f"Start testing on {test_loader.size} images...")

    # 4. 初始化指标列表
    MAE_list, Recall_list, Precision_list, Dice_list = [], [], [], []
    pred_list, gt_list = [], []

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()

            # GT 处理
            gt_np = np.asarray(gt, np.float32)
            gt_np = cv2.resize(gt_np, (opt.testsize, opt.testsize), interpolation=cv2.INTER_NEAREST)
            gt_np = gt_np / (gt_np.max() + 1e-8)

            # 模型推理
            image = image.to(device)
            predicts = model(image)

            # 预测处理
            res = predicts[0]
            res = F.interpolate(res, size=(opt.testsize, opt.testsize), mode='bilinear', align_corners=False)
            res = res.sigmoid().squeeze().cpu().numpy()

            pred_list.append(res)
            gt_list.append(gt_np)

            # 可视化保存
            save_img = ((res > 0.5) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(save_path, name), save_img)

            # 指标计算
            pred_binary = (res >= 0.5).astype(np.float32)
            pred_binary_inverse = (pred_binary == 0).astype(np.float32)
            gt_binary = (gt_np >= 0.5).astype(np.float32)
            gt_binary_inverse = (gt_binary == 0).astype(np.float32)

            MAE = np.mean(np.abs(pred_binary - gt_binary))
            TP = np.sum(pred_binary * gt_binary)
            FP = np.sum(pred_binary * gt_binary_inverse)
            TN = np.sum(pred_binary_inverse * gt_binary_inverse)
            FN = np.sum(pred_binary_inverse * gt_binary)

            if TP == 0: TP = 1.0
            Recall = TP / (TP + FN)
            Precision = TP / (TP + FP)
            Dice = 2 * Precision * Recall / (Precision + Recall)

            MAE_list.append(MAE)
            Recall_list.append(Recall)
            Precision_list.append(Precision)
            Dice_list.append(Dice)

    # ---------- 计算 F1、Recall、Precision、MAE ----------
    mae = np.mean(MAE_list)
    recall = np.mean(Recall_list)
    precision = np.mean(Precision_list)
    f1 = np.mean(Dice_list)

    # ---------- mIoU 阈值扫描 ----------
    thresh_step = 0.01
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        thresh_iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt >= 0.5).astype(np.uint8)
            pred_img = (pred >= thresh).astype(np.uint8)
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) == 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0) / 2
            thresh_iou_list.append(iou)
        final_iou.append(np.mean(thresh_iou_list))

    miou = np.max(final_iou)


    print("\n" + "=" * 50)
    print(f"Test Results on {test_loader.size} images:")
    print(f"mIoU      : {miou:.4f}")
    print(f"F1 / Dice : {f1:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"MAE       : {mae:.4f}")
    print("=" * 50)


