import torch
import torch.nn.functional as F
import numpy as np


def adjusted_rand_index(true_ids, pred_ids, ignore_background=True):
    """Computes the adjusted Rand index (ARI), a clustering similarity score.

    Code borrowed from https://github.com/google-research/slot-attention-video/blob/e8ab54620d0f1934b332ddc09f1dba7bc07ff601/savi/lib/metrics.py#L111

    Args:
        true_ids: An integer-valued array of shape
            [batch_size, seq_len, H, W]. The true cluster assignment encoded
            as integer ids.
        pred_ids: An integer-valued array of shape
            [batch_size, seq_len, H, W]. The predicted cluster assignment
            encoded as integer ids.
        ignore_background: Boolean, if True, then ignore all pixels where
            true_ids == 0 (default: False).

    Returns:
        ARI scores as a float32 array of shape [batch_size].
    """
    if len(true_ids.shape) == 3:
        true_ids = true_ids.unsqueeze(1)
    if len(pred_ids.shape) == 3:
        pred_ids = pred_ids.unsqueeze(1)

    true_oh = F.one_hot(true_ids).float()
    pred_oh = F.one_hot(pred_ids).float()

    if ignore_background:
        true_oh = true_oh[..., 1:]  # Remove the background row.

    N = torch.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
    A = torch.sum(N, dim=-1)  # row-sum  (batch_size, c)
    B = torch.sum(N, dim=-2)  # col-sum  (batch_size, k)
    num_points = torch.sum(A, dim=1)

    rindex = torch.sum(N * (N - 1), dim=[1, 2])
    aindex = torch.sum(A * (A - 1), dim=1)
    bindex = torch.sum(B * (B - 1), dim=1)
    expected_rindex = (
        aindex * bindex / torch.clamp(num_points * (num_points - 1), min=1)
    )
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator != 0, ari, torch.tensor(1.0).type_as(ari))


def ARI_metric(x, y):
    """x/y: [B, H, W], both are seg_masks after argmax."""
    assert "int" in str(x.dtype)
    assert "int" in str(y.dtype)
    return adjusted_rand_index(x, y, ignore_background=False).mean().item()


def fARI_metric(x, y):
    """x/y: [B, H, W], both are seg_masks after argmax."""
    assert "int" in str(x.dtype)
    assert "int" in str(y.dtype)
    return adjusted_rand_index(x, y, ignore_background=True).mean().item()


def basic_miou(gt_mask, pred_mask):
    """both mask: [H*W] after argmax, 0 is gt background index."""
    gt_mask_0s = torch.sum(gt_mask, dim=1) == 0

    intersect = torch.sum(gt_mask * pred_mask, dim=1)
    union = torch.sum(gt_mask | pred_mask, dim=1)

    iou = intersect / (union + 1e-8)

    # If the entire mask is 0s, then the IoU is 1
    iou[gt_mask_0s] = 1

    iou = iou.detach().cpu().numpy()

    return iou  # iou[row_ind, col_ind].sum() / float(N)


def miou_metric(gt_mask, pred_mask, reduce=True):
    """both mask: [B, H, W], both are seg_masks after argmax."""
    assert "int" in str(gt_mask.dtype)

    """ print('GT MASK ', gt_mask.shape, gt_mask.min(), gt_mask.max())
    print('PRED MASK ', pred_mask.shape)  """

    gt_mask, pred_mask = gt_mask.flatten(1, 2), pred_mask.flatten(1, 2)
    ious = basic_miou(gt_mask, pred_mask)
    if reduce:
        return ious.mean()
    return ious


@torch.no_grad()
def get_binary_mask_from_rgb(x):
    """
    B,C,H,W
    """

    x = x.clamp(0, 1)
    x = torch.mean(x, dim=-3)

    x[x < 0.1] = 0
    x[x >= 0.1] = 1

    x = x.long().squeeze(-3)

    return x


def get_metrics(gt, pred, return_bins=False):
    bin_gt = get_binary_mask_from_rgb(gt)
    bin_pred = get_binary_mask_from_rgb(pred)

    metric_dict = {}
    metric_dict["ARI"] = ARI_metric(bin_gt, bin_pred)
    metric_dict["fARI"] = fARI_metric(bin_gt, bin_pred)
    metric_dict["mIoU"] = miou_metric(bin_gt, bin_pred)

    if return_bins:
        return metric_dict, bin_gt, bin_pred
    else:
        return metric_dict


import cv2


def save_bins_to_image(
    bin_masks, pathprefix, labels=["GT", "SAVi", "CarFormer"], overlays=None
):
    # Bin masks:
    # list of BxHxW binary masks
    # visualize B images, each of which is of size len(bin_masks)*H x W
    # we assume that index 0 is GT
    # Get pairwise GT
    assert len(bin_masks) <= 3, "Only 3 masks supported"
    gts = bin_masks[0].unsqueeze(1).expand(-1, len(bin_masks), -1, -1)

    preds = torch.stack(bin_masks, dim=1)

    # Get pairwise mious
    pair_mious = miou_metric(
        gts.flatten(0, 1), preds.flatten(0, 1), reduce=False
    ).reshape(gts.shape[0], len(bin_masks), -1)

    # resize all bin_masks and overlays to twice the size
    bin_masks = [
        F.interpolate(x.unsqueeze(1).float() * 255, scale_factor=(2, 2)).squeeze(1)
        for x in bin_masks
    ]
    if overlays is not None:
        overlays = [
            F.interpolate(x.unsqueeze(1).float() * 255, scale_factor=(2, 2)).squeeze(1)
            for x in overlays
        ]

    white_line = np.ones((bin_masks[0].shape[1], 1, 3), dtype=np.uint8) * 255

    combined_mask = np.zeros((*bin_masks[0][0].shape, 3), dtype=np.uint8)

    for i in range(bin_masks[0].shape[0]):
        to_concat = []
        for j in range(len(bin_masks)):
            miou = pair_mious[i, j].item()

            mask = bin_masks[j][i].cpu().numpy()

            combined_mask[:, :, 2 - j] = mask

            if overlays is not None:
                overlay = overlays[j][i].cpu().numpy()
                mask = cv2.addWeighted(
                    mask.astype(np.uint8), 0.7, overlay.astype(np.uint8), 0.3, 0
                )

            # Convert to RGB
            mask = np.stack([mask, mask, mask], axis=-1)

            # Write MIOU centered in the top of the image. Images are 192x192 so we write in the top 20 pixels
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            fontColor = (0, 0, 255)
            lineType = 2

            cv2.putText(
                combined_mask,
                labels[j],
                (20, 28 + 28 * j),
                font,
                fontScale * 1.4,
                tuple([int(255 * ((2 - j) == x)) for x in range(3)]),
                lineType,
            )

            cv2.putText(
                mask,
                "{:.2f}".format(miou),
                (20, 40),
                font,
                fontScale * 2,
                fontColor,
                lineType,
            )

            # cv2.putText(
            #     mask,
            #     labels[j],
            #     (20, 40),
            #     font,
            #     fontScale,
            #     fontColor,
            #     lineType,
            # )

            to_concat.append(mask)
            to_concat.append(white_line)

        to_concat.append(combined_mask)

        img = np.concatenate(to_concat, axis=1)  # LH x W x 3

        # Write the MIOUs with
        cv2.imwrite(pathprefix + "_" + str(i) + ".png", img)


def save_rgbs_to_image(rgb_masks, pathprefix, labels=["GT", "SAVi", "CarFormer"]):
    # Bin masks:
    # list of BxHxW binary masks
    # visualize B images, each of which is of size len(bin_masks)*H x W
    # we assume that index 0 is GT
    # Get pairwise GT
    assert len(rgb_masks) <= 3, "Only 3 masks supported"

    # resize all bin_masks and overlays to twice the size
    # for rgb in rgb_masks:
    #     print(rgb.shape)
    # import ipdb

    # ipdb.set_trace()
    bin_masks = [
        F.interpolate(
            (x.squeeze(0).float().clamp(0, 1) * 0.5 + 0.5) * 255, scale_factor=(1, 2, 2)
        ).squeeze()
        for x in rgb_masks
    ]

    # Move channel to the end
    bin_masks = [x.permute(0, 2, 3, 1) for x in bin_masks]

    white_line = np.ones((bin_masks[0].shape[1], 1, 3), dtype=np.uint8) * 255

    for i in range(bin_masks[0].shape[0]):
        to_concat = []
        for j in range(len(bin_masks)):
            mask = bin_masks[j][i].cpu().numpy()

            # Convert to RGB
            # mask = np.stack([mask, mask, mask], axis=-1)

            to_concat.append(mask)
            to_concat.append(white_line)

        img = np.concatenate(to_concat[:-1], axis=1)  # LH x W x 3

        # Write the MIOUs with
        cv2.imwrite(pathprefix + "_rgb_" + str(i) + ".png", img)


def save_bins_to_image_nomiou(
    bin_masks,
    pathprefix,
    labels=["GT", "Input", "Pred"],
    overlays=None,
    return_image=False,
):
    # Bin masks:
    # list of BxHxW binary masks
    # visualize B images, each of which is of size len(bin_masks)*H x W
    # we assume that index 0 is GT
    # Get pairwise GT
    # assert len(bin_masks) <= 3, "Only 3 masks supported"
    # resize all bin_masks and overlays to twice the size
    bin_masks = [
        F.interpolate(x.unsqueeze(1).float() * 255, scale_factor=(2, 2)).squeeze(1)
        for x in bin_masks
    ]

    if overlays is not None:
        overlays = [
            F.interpolate(x.unsqueeze(1).float() * 255, scale_factor=(2, 2)).squeeze(1)
            for x in overlays
        ]

    white_line = np.ones((bin_masks[0].shape[1], 1, 3), dtype=np.uint8) * 255

    combined_mask = np.zeros((*bin_masks[0][0].shape, 3), dtype=np.uint8)

    for i in range(bin_masks[0].shape[0]):
        to_concat = []
        for j in range(len(bin_masks)):
            mask = bin_masks[j][i].cpu().numpy()

            combined_mask[:, :, 2 - j] = mask

            if overlays is not None:
                overlay = overlays[j][i].cpu().numpy()
                mask = cv2.addWeighted(
                    mask.astype(np.uint8), 0.7, overlay.astype(np.uint8), 0.3, 0
                )

            # Convert to RGB
            mask = np.stack([mask, mask, mask], axis=-1)

            # Write MIOU centered in the top of the image. Images are 192x192 so we write in the top 20 pixels
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            fontColor = (0, 0, 255)
            lineType = 2

            cv2.putText(
                combined_mask,
                labels[j],
                (20, 28 + 28 * j),
                font,
                fontScale * 1.4,
                tuple([int(255 * ((2 - j) == x)) for x in range(3)]),
                lineType,
            )

            to_concat.append(mask)
            to_concat.append(white_line)

        to_concat.append(combined_mask)

        img = np.concatenate(to_concat, axis=1)  # LH x W x 3

        # Write the MIOUs with
        cv2.imwrite(f"{i:0>4d}_{pathprefix}.png", img)

    if return_image:
        return img
