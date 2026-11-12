import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from torch.autograd import Variable
import gc
import os
import argparse
from lib.HPE import HPEModel
from utils.dataloader import PolypDataset, get_loader, test_dataset,val_dataset
from utils.loss import DeepSupervisionLoss
from utils.utils import AvgMeter, clip_gradient
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='0'

def train(train_loader, model, optimizer, epoch, criteria_loss):
    model.train()
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, name = pack
        images = images.to(device)
        gts = gts.to(device)

        predicts = model(images)
        loss = criteria_loss(predicts, gts)

        writer.add_scalar("Loss/train", loss, epoch * total_step + i)

        loss.backward()
        clip_gradient(optimizer, opt.grad_norm)
        optimizer.step()

        loss_record.update(loss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}'.format(
                datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    save_path = os.path.join(opt.save_path, opt.model, 'checkpoints')
    os.makedirs(save_path, exist_ok=True)

    if (epoch + 1) % 50 == 0:
        ckpt = os.path.join(save_path, 'FreqNet-%d.pth' % epoch)
        torch.save(model.state_dict(), ckpt)
        print('[Saving Snapshot:]', ckpt)


def val(test_loader, model, epoch):
    global best_f1, best_epoch
    model.eval()

    pred_list = []
    gt_list = []

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()

            gt_np = np.asarray(gt, np.float32)
            gt_np = cv2.resize(
                gt_np,
                (opt.trainsize, opt.trainsize),
                interpolation=cv2.INTER_NEAREST
            )
            gt_np = gt_np / (gt_np.max() + 1e-8)
            gt_list.append(gt_np)

            image = image.to(device)
            result = model(image)

            res = result[0]
            res = F.interpolate(
                res,
                size=(opt.trainsize, opt.trainsize),
                mode='bilinear',
                align_corners=False
            )
            res = res.sigmoid().squeeze().cpu().numpy()
            pred_list.append(res)

    # ================= F1、Recall、Precision、MAE =================
    MAE_list = []
    Recall_list = []
    Precision_list = []
    F1_list = []

    for pred, gt in zip(pred_list, gt_list):
        pred_binary = (pred >= 0.5).astype(np.float32)
        pred_binary_inverse = (pred_binary == 0).astype(np.float32)

        gt_binary = (gt >= 0.5).astype(np.float32)
        gt_binary_inverse = (gt_binary == 0).astype(np.float32)

        MAE = np.mean(np.abs(pred_binary - gt_binary))

        TP = np.sum(pred_binary * gt_binary)
        FP = np.sum(pred_binary * gt_binary_inverse)
        TN = np.sum(pred_binary_inverse * gt_binary_inverse)
        FN = np.sum(pred_binary_inverse * gt_binary)

        if TP == 0:
            TP = 1.0

        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        F1 = 2 * Precision * Recall / (Precision + Recall)

        MAE_list.append(MAE)
        Recall_list.append(Recall)
        Precision_list.append(Precision)
        F1_list.append(F1)

    mae = np.mean(MAE_list)
    recall = np.mean(Recall_list)
    precision = np.mean(Precision_list)
    f1 = np.mean(F1_list)

    # ================= mIoU =================
    thresh_step = 0.01
    final_iou = []

    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []

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

            iou_list.append(iou)

        final_iou.append(np.mean(iou_list))

    miou = np.max(final_iou)

    # ================= 记录日志 =================
    save_path = os.path.join(
        opt.save_path,
        opt.model,
        'checkpoints'
    )
    os.makedirs(save_path, exist_ok=True)

    writer.add_scalar('MAE', mae, global_step=epoch)
    writer.add_scalar('mIoU', miou, global_step=epoch)
    writer.add_scalar('F1', f1, global_step=epoch)
    writer.add_scalar('Recall', recall, global_step=epoch)
    writer.add_scalar('Precision', precision, global_step=epoch)

    print(
        'Epoch: {}, mIoU: {:.4f}, F1: {:.4f}, Recall: {:.4f}, '
        'Precision: {:.4f}, MAE: {:.4f}, bestF1: {:.4f}, bestEpoch: {}.'
        .format(
            epoch, miou, f1, recall, precision, mae, best_f1, best_epoch
        )
    )

    # ================= 按 F1 保存最佳模型 =================
    if epoch == 0:
        best_f1 = f1
        best_epoch = 0
        torch.save(
            model.state_dict(),
            os.path.join(save_path, "best_epoch.pth")
        )
    else:
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "best_epoch.pth")
            )
            print(
                'Save state_dict successfully! Best epoch:{}.'
                .format(epoch)
            )

    # ================= 清理内存 =================
    del pred_list
    del gt_list
    gc.collect()


if __name__ == '__main__':
    

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument('--grad_norm', type=float, default=0.5, help='gradient clipping norm')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mt', type=float, default=0.9)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--trainsize', type=int,
                        default=512, help='training dataset size')
    # parser.add_argument('--train_path', type=str,
    #                     default='', help='path to train dataset')
    parser.add_argument('--train_path', type=str,
                        default='/home/weimengru/dataset/crack500/data',
                        help='path to dataset root (contains images/, labels/, annotations/)')
    parser.add_argument('--model', type=str,
                        default='HFFI')
    parser.add_argument('--save_path', type=str,
                        default='/home/weimengru/dataset/')
    parser.add_argument("--mgpu", type=str, default="false", choices=["true", "false"])
    opt = parser.parse_args()
    # writer = SummaryWriter(opt.save_path +opt.model + "/tensorboard/")
    writer = SummaryWriter(os.path.join(opt.save_path, opt.model, "tensorboard"))
    # model = HFEANetModel()
    model = HPEModel()

    if opt.mgpu == "true" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    params = model.parameters()

    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.epoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    criteria_loss = DeepSupervisionLoss()

    data_root = opt.train_path
    images_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')
    train_txt = os.path.join(data_root, 'annotations', 'train.txt')
    test_txt = os.path.join(data_root, 'annotations', 'test.txt')

    train_loader = get_loader(images_dir, labels_dir, train_txt,
                              batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    val_loader = val_dataset(images_dir, labels_dir, test_txt, testsize=opt.trainsize)


    best_f1 = 0.0
    best_epoch = 0
    print(torch.cuda.get_device_name(0))
    print("#"*20, "Start Training", "#"*20)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    for epoch in range(0, opt.epoch):
        train(train_loader, model, optimizer, epoch, criteria_loss)
        val(val_loader,model,epoch)
        scheduler.step()

    writer.close()

