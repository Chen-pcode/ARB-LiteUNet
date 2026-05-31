import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
import os
from PIL import Image


def _to_uint8_image(img):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return (img * 255).clip(0, 255).astype(np.uint8)


def _to_rgb_mask(mask):
    mask = np.squeeze(mask)
    mask = (mask > 0.5).astype(np.uint8) * 255
    return np.stack([mask, mask, mask], axis=-1)


def _save_prediction(img, mask, pred, idx, save_path, threshold=0.5, test_data_name=None):
    os.makedirs(save_path, exist_ok=True)
    image_np = _to_uint8_image(img)
    mask_np = _to_rgb_mask(mask)
    pred_np = _to_rgb_mask(pred >= threshold)
    canvas = np.concatenate([image_np, mask_np, pred_np], axis=1)
    prefix = f"{test_data_name}_" if test_data_name is not None else ""
    Image.fromarray(canvas).save(os.path.join(save_path, f"{prefix}{idx}.png"))


def _calculate_hd95_list(preds_list, gts_list, threshold=0.5):
    try:
        from medpy.metric.binary import hd95
    except Exception:
        return 0.0

    scores = []
    for batch_pred, batch_gt in zip(preds_list, gts_list):
        for i in range(batch_pred.shape[0]):
            p = (batch_pred[i] >= threshold).astype(int)
            g = (batch_gt[i] >= threshold).astype(int)
            if np.sum(p) > 0 and np.sum(g) > 0:
                try:
                    scores.append(hd95(p, g))
                except Exception:
                    pass
    return float(np.mean(scores)) if scores else 0.0


def _collect_metrics(preds, gts, threshold):
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    y_pre = np.where(preds >= threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre)
    if confusion.size == 1:
        if y_true.sum() == 0 and y_pre.sum() == 0:
            TN, FP, FN, TP = confusion[0, 0], 0, 0, 0
        else:
            TN, FP, FN, TP = 0, 0, 0, confusion[0, 0]
    else:
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "dice": f1_or_dsc,
        "miou": miou,
        "confusion": confusion,
    }


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, writer=None, scaler=None):
    model.train()
    loss_list = []
    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        if writer is not None:
            writer.add_scalar("loss", loss.item(), global_step=epoch * len(train_loader) + iter)
        if iter % config.print_interval == 0:
            log_info = f"train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}"
            print(log_info)
            logger.info(log_info)

    scheduler.step()
    return float(np.mean(loss_list))


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    deep_pred_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()
            deep_masks, deep_boundaries, out = model(img)
            loss = criterion((deep_masks, deep_boundaries, out), msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            out_np = out.squeeze(1).cpu().detach().numpy()
            preds.append(out_np)
            deep_pred_list.append(deep_masks[0].squeeze(1).cpu().detach().numpy())

    metrics = _collect_metrics(preds, gts, config.threshold)
    avg_hd95 = _calculate_hd95_list(preds, gts, config.threshold)
    if epoch % config.val_interval == 0:
        log_info = (
            f"val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {metrics['miou']}, "
            f"dice: {metrics['dice']}, accuracy: {metrics['accuracy']}, "
            f"specificity: {metrics['specificity']}, sensitivity: {metrics['sensitivity']}, hd95: {avg_hd95}, "
            f"confusion_matrix: {metrics['confusion']}"
        )
        print(log_info)
        logger.info(log_info)
    else:
        log_info = f"val epoch: {epoch}, loss: {np.mean(loss_list):.4f}"
        print(log_info)
        logger.info(log_info)
    return float(np.mean(loss_list))


def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None, save_root=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()
            deep_masks, deep_boundaries, out = model(img)
            loss = criterion((deep_masks, deep_boundaries, out), msk)
            loss_list.append(loss.item())
            msk_np = msk.squeeze(1).cpu().detach().numpy()
            out_np = out.squeeze(1).cpu().detach().numpy()
            gts.append(msk_np)
            preds.append(out_np)
            if save_root is not None and i % config.save_interval == 0:
                _save_prediction(img, msk_np, out_np, i, save_root, config.threshold, test_data_name=test_data_name)

    metrics = _collect_metrics(preds, gts, config.threshold)
    avg_hd95 = _calculate_hd95_list(preds, gts, config.threshold)
    if test_data_name is not None:
        logger.info(f"test_datasets_name: {test_data_name}")
    log_info = (
        f"test of best model, loss: {np.mean(loss_list):.4f}, miou: {metrics['miou']}, "
        f"dice: {metrics['dice']}, accuracy: {metrics['accuracy']}, specificity: {metrics['specificity']}, "
        f"sensitivity: {metrics['sensitivity']}, hd95: {avg_hd95}, confusion_matrix: {metrics['confusion']}"
    )
    print(log_info)
    logger.info(log_info)
    return float(np.mean(loss_list))
