import torch
from torch import nn
import torch.nn.functional as F



def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


'''for ms3 and s4 task'''
def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss



def overlap_loss(inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    batch_seg_token_count: int):
    if num_masks == 0:
        return inputs.sum() * 0
    print('batch_seg_token_count [1]: ',batch_seg_token_count) # tensor(1, device='npu:1')
    batch_seg_token_count = batch_seg_token_count.cumsum(-1) 
    print('batch_seg_token_count [2]: ',batch_seg_token_count) # # tensor(1, device='npu:1')
    batch_seg_token_count = torch.cat(
            [torch.zeros(1).long().cuda(), batch_seg_token_count], dim=0
        )
    loss = 0

    for i in range(len(batch_seg_token_count) -1):
        start_i = batch_seg_token_count[i]
        end_i = batch_seg_token_count[i+1]
        assert end_i <= len(targets), (targets.shape, batch_seg_token_count)
        question_inputs = inputs[start_i:end_i]
        question_targets = targets[start_i:end_i]
        if len(question_targets) == 0:
            continue
        n, h, w = question_inputs.shape
        all_targets = torch.zeros_like(question_targets[0]).bool()
        for target in question_targets:
            all_targets = (all_targets | target.bool())
        bg_area = all_targets < 0
        bg_area = bg_area[None].repeat(n, 1, 1)

        overlap_area = (question_inputs > 0).sum(dim=0)
        overlap_area = overlap_area >= 2

        overlap_area = overlap_area[None].repeat(n, 1, 1)
        weight = torch.ones_like(question_inputs)
        weight[~overlap_area] = 0

        q_loss = F.binary_cross_entropy_with_logits(question_inputs, question_targets, weight=weight, reduction="none")
        q_loss = q_loss.flatten(1, 2).mean(1).sum() 
        loss = loss + q_loss
    loss = loss / (num_masks + 1e-8)
    return loss


'''for avss task'''

def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total ten frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total ten frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1)

    loss = nn.CrossEntropyLoss(reduction='none')(
        pred_mask, ten_gt_masks)  # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1)  # [bs*10]
    # loss = loss * gt_temporal_mask_flag  # [bs*10]
    # loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
    loss = torch.sum(loss)

    return loss




