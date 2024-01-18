# Modified from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
FastInst criterion.
"""

import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size
from torch import nn

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from os.path import join
import numpy as np

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: number of masks
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


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
        num_masks: number of masks
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


# class SetCriterion(nn.Module):
#     """This class computes the loss for DETR.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth boxes and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
#     """
#
#     def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
#                  num_points, oversample_ratio, importance_sample_ratio):
#         """Create the criterion.
#         Parameters:
#             num_classes: number of object categories, omitting the special no-object category
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losses and as values their relative weight.
#             eos_coef: relative classification weight applied to the no-object category
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.eos_coef = eos_coef
#         self.losses = losses
#         empty_weight = torch.ones(self.num_classes + 1)
#         empty_weight[-1] = self.eos_coef
#         self.register_buffer("empty_weight", empty_weight)
#
#         # pointwise mask loss parameters
#         self.num_points = num_points
#         self.oversample_ratio = oversample_ratio
#         self.importance_sample_ratio = importance_sample_ratio
#
#     def loss_labels(self, outputs, targets, indices, num_masks):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert "pred_logits" in outputs
#         src_logits = outputs["pred_logits"].float()
#
#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(
#             src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
#         )
#
#         target_classes[idx] = target_classes_o
#
#         loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
#         losses = {"loss_ce": loss_ce}
#         return losses
#
#     def loss_masks(self, outputs, targets, indices, num_masks):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#         targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#         """
#         assert "pred_masks" in outputs
#
#         src_idx = self._get_src_permutation_idx(indices)
#         tgt_idx = self._get_tgt_permutation_idx(indices)
#         src_masks = outputs['pred_masks']
#         src_masks = src_masks[src_idx]
#         masks = [t["masks"] for t in targets]
#         target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
#         target_masks = target_masks.to(src_masks)
#         target_masks = target_masks[tgt_idx]
#
#         # No need to upsample predictions as we are using normalized coordinates :)
#         # N x 1 x H x W
#         src_masks = src_masks[:, None]
#         target_masks = target_masks[:, None]
#
#         with torch.no_grad():
#             # sample point_coords
#             point_coords = get_uncertain_point_coords_with_randomness(
#                 src_masks,
#                 lambda logits: calculate_uncertainty(logits),
#                 self.num_points,
#                 self.oversample_ratio,
#                 self.importance_sample_ratio,
#             )
#             # get gt labels
#             point_labels = point_sample(
#                 target_masks,
#                 point_coords,
#                 align_corners=False,
#             ).squeeze(1)
#
#         point_logits = point_sample(
#             src_masks,
#             point_coords,
#             align_corners=False,
#         ).squeeze(1)
#
#         losses = {
#             "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
#             "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
#         }
#
#         del src_masks
#         del target_masks
#         return losses
#
#     def loss_proposals(self, output_proposals, targets, indices):
#         assert "proposal_cls_logits" in output_proposals
#
#         proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]
#         proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()  # b, c, hw
#
#         target_classes = self.num_classes * torch.ones([proposal_cls_logits.shape[0],
#                                                         proposal_size[0] * proposal_size[1]],
#                                                        device=proposal_cls_logits.device)
#         target_classes = target_classes.to(torch.int64)
#
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         idx = self._get_src_permutation_idx(indices)
#         target_classes[idx] = target_classes_o
#
#         loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
#         losses = {"loss_proposal": loss_proposal}
#
#         return losses
#
#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx
#
#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx
#
#     def get_loss(self, loss, outputs, targets, indices, num_masks):
#         loss_map = {
#             'labels': self.loss_labels,
#             'masks': self.loss_masks,
#         }
#         assert loss in loss_map, f"do you really want to compute {loss} loss?"
#         return loss_map[loss](outputs, targets, indices, num_masks)

    # def forward(self, outputs, targets):
    #     """This performs the loss computation.
    #     Parameters:
    #          outputs: dict of tensors, see the output specification of the model for the format
    #          targets: list of dicts, such that len(targets) == batch_size.
    #                   The expected keys in each dict depends on the losses applied, see each loss' doc
    #     """
    #     # Compute proposal loss
    #     proposal_loss_dict = {}
    #     if outputs.get("proposal_cls_logits") is not None:
    #         output_proposals = {"proposal_cls_logits": outputs.pop("proposal_cls_logits")}
    #         indices = self.matcher(output_proposals, targets)
    #         proposal_loss_dict = self.loss_proposals(output_proposals, targets, indices)
    #
    #     # Compute the main output loss
    #     outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
    #
    #     # Retrieve the matching between the outputs of the last layer and the targets
    #     if outputs_without_aux.get("pred_matching_indices") is not None:
    #         indices = outputs_without_aux["pred_matching_indices"]
    #     else:
    #         indices = self.matcher(outputs_without_aux, targets)
    #
    #     # Compute the average number of target boxes across all nodes, for normalization purposes
    #     num_masks = sum(len(t["labels"]) for t in targets)
    #     num_masks = torch.as_tensor(
    #         [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
    #     )
    #     if is_dist_avail_and_initialized():
    #         torch.distributed.all_reduce(num_masks)
    #     num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
    #
    #     # Compute all the requested losses
    #     losses = {}
    #     for loss in self.losses:
    #         losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
    #
    #     # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
    #     if "aux_outputs" in outputs:
    #         for i, aux_outputs in enumerate(outputs["aux_outputs"]):
    #             if aux_outputs.get("pred_matching_indices") is not None:
    #                 indices = aux_outputs["pred_matching_indices"]
    #             else:
    #                 indices = self.matcher(aux_outputs, targets)
    #             for loss in self.losses:
    #                 l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
    #                 l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
    #                 losses.update(l_dict)
    #
    #     losses.update(proposal_loss_dict)
    #
    #     return losses



    # def __repr__(self):
    #     head = "Criterion " + self.__class__.__name__
    #     body = [
    #         "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
    #         "losses: {}".format(self.losses),
    #         "weight_dict: {}".format(self.weight_dict),
    #         "num_classes: {}".format(self.num_classes),
    #         "eos_coef: {}".format(self.eos_coef),
    #         "num_points: {}".format(self.num_points),
    #         "oversample_ratio: {}".format(self.oversample_ratio),
    #         "importance_sample_ratio: {}".format(self.importance_sample_ratio),
    #     ]
    #     _repr_indent = 4
    #     lines = [head] + [" " * _repr_indent + line for line in body]
    #     return "\n".join(lines)

import copy

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, eos_coef, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        # No need to upsample predictions as we are using normalized coordinates :)
                # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def merge_indices(self, indices, size):
        all_indices = []
        assert len(indices[0]) == len(indices[1])
        for handi, obji in zip(indices[0], indices[1]):
            new_tuple = (
            torch.as_tensor([*handi[0], size // 2 + obji[0]]), torch.as_tensor([*handi[1], 2 if len(handi[1]) == 2 else 1]))
            all_indices.append(new_tuple)
        return all_indices
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # indices = self.merge_indices(indices, src_logits.shape[1])
        idx = self._get_src_permutation_idx(indices)
        try:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).cuda()
        except:
            import pdb; pdb.set_trace()
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device).cuda()
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
    #     """Classification loss (NLL)
    #     targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    #     """
    #     assert "pred_logits" in outputs
    #     src_logits = outputs["pred_logits"].float()
    #
    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).cuda()
    #     target_classes = torch.full(
    #         src_logits.shape[:2], self.num_classes + 1, dtype=torch.int64, device=src_logits.device
    #     ).cuda()
    #
    #     target_classes[idx] = target_classes_o
    #
    #     loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
    #     losses = {"loss_ce": loss_ce}
    #     return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_hand_2dkeypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        _, _, key_left, key_right = outputs['pred_keypoints']
        # src_keypoints = outputs['pred_keypoints'][idx]
        key_left = key_left.view(key_left.shape[0], key_left.shape[1], 21, 3)
        key_right = key_right.view(key_right.shape[0], key_right.shape[1], 21, 3)
        key_left = key_left[..., :2].flatten(2)
        key_right = key_right[..., :2].flatten(2)
        src_keypoints_left = key_left[idx]
        src_keypoints_right = key_right[idx]
        # tgt_indices = self.merge_indices(indices, outputs['pred_logits'].shape[1])
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()

        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        left_hand_idx = (target_labels == 9).to(torch.bool).cuda()
        right_hand_idx = (target_labels == 10).to(torch.bool).cuda()
        # hand_cal_idx = ((target_labels == 9).to(torch.int64) + (target_labels==10).to(torch.int64)) == 1
        # obj_cal_idx = ((target_labels == 9).to(torch.int64) + (target_labels==10).to(torch.int64)) == 0

        target_6Dpose = torch.cat([t['obj6Dpose'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()
        # target_6Dpose_flipcheck = torch.cat([t['obj6Dpose'][i] * (t['flip_prob'] > 0.5) for t, (_, i) in zip(targets, indices)], dim=0)
        # FLIP 되지 않은것들
        hand_cal_idx = target_6Dpose.sum(dim=1) == 0

        # loss_handkey = F.l1_loss(src_keypoints[hand_cal_idx], target_keypoints[hand_cal_idx].view(-1, 63),
        #                          reduction='none')
        # import pdb; pdb.set_trace()
        left_loss_handkey = F.l1_loss(src_keypoints_left[left_hand_idx], target_keypoints[left_hand_idx][..., :2].flatten(1),
                                      reduction='none')
        right_loss_handkey = F.l1_loss(src_keypoints_right[right_hand_idx],
                                       target_keypoints[right_hand_idx][..., :2].flatten(1),
                                       reduction='none')

        loss_handkey = left_loss_handkey.sum() + right_loss_handkey.sum()

        losses = {}
        losses['loss_hand_2dkeypoint'] = (loss_handkey / hand_cal_idx.sum().item()) / 21
        return losses
    def loss_hand_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        key_left, key_right, _, _ = outputs['pred_keypoints']
        # src_keypoints = outputs['pred_keypoints'][idx]
        src_keypoints_left = key_left[idx]
        src_keypoints_right = key_right[idx]
        # tgt_indices = self.merge_indices(indices, outputs['pred_logits'].shape[1])
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()

        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        left_hand_idx = (target_labels == 9).to(torch.bool).cuda()
        right_hand_idx = (target_labels == 10).to(torch.bool).cuda()
        # hand_cal_idx = ((target_labels == 9).to(torch.int64) + (target_labels==10).to(torch.int64)) == 1
        # obj_cal_idx = ((target_labels == 9).to(torch.int64) + (target_labels==10).to(torch.int64)) == 0

        target_6Dpose = torch.cat([t['obj6Dpose'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()
        # target_6Dpose_flipcheck = torch.cat([t['obj6Dpose'][i] * (t['flip_prob'] > 0.5) for t, (_, i) in zip(targets, indices)], dim=0)
        # FLIP 되지 않은것들
        hand_cal_idx = target_6Dpose.sum(dim=1) == 0

        # loss_handkey = F.l1_loss(src_keypoints[hand_cal_idx], target_keypoints[hand_cal_idx].view(-1, 63),
        #                          reduction='none')
        left_loss_handkey = F.l1_loss(src_keypoints_left[left_hand_idx], target_keypoints[left_hand_idx].view(-1, 63),
                                 reduction='none')
        right_loss_handkey = F.l1_loss(src_keypoints_right[right_hand_idx], target_keypoints[right_hand_idx].view(-1, 63),
                                 reduction='none')



        loss_handkey = left_loss_handkey.sum() + right_loss_handkey.sum()

        losses = {}
        losses['loss_hand_keypoint'] = (loss_handkey / hand_cal_idx.sum().item()) / 21
        # losses['loss_hand_keypoint'] =  torch.mean(torch.sum((target_keypoints[hand_cal_idx].view(-1, 63) - src_keypoints[hand_cal_idx])**2, dim=-1))

        return losses
    def loss_delta_hand(self, outputs, targets, indices, num_boxes):
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        target_6Dpose = torch.cat([t['obj6Dpose'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()
        # target_6Dpose_flipcheck = torch.cat([t['obj6Dpose'][i] * (t['flip_prob'] > 0.5) for t, (_, i) in zip(targets, indices)], dim=0)
        # FLIP 되지 않은것들
        hand_cal_idx = target_6Dpose.sum(dim=1) == 0
        src_keypoints = outputs['pred_keypoints'][idx][hand_cal_idx]
        #paths = np.array([[join(*t['img_path'].split("/")[:-1])]*t['obj6Dpose'][i].shape[0] for t, (_, i) in zip(targets, indices)]).flatten()[hand_cal_idx.detach().cpu()].tolist()
        delta_loss_pred = []
        for i in range(src_keypoints.shape[0] - 1 , 0, -1):
            for j in range(i % 2, i, 2):
                #if paths[i] == paths[j]:
                delta_loss_pred.append(torch.abs(src_keypoints[i] - src_keypoints[j]))

        delta_loss_pred = torch.stack(delta_loss_pred)
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()

        delta_loss_target = []
        target_keypoints = target_keypoints[hand_cal_idx].view(-1, 63)
        for i in range(target_keypoints.shape[0] - 1, 0, -1):
            for j in range(i % 2, i, 2):
                #if paths[i] == paths[j]:
                delta_loss_target.append(torch.abs(target_keypoints[i] - target_keypoints[j]))
        delta_loss_target = torch.stack(delta_loss_target)


        delta_loss_hand = F.l1_loss(delta_loss_pred, delta_loss_target,
                                 reduction='none')

        losses = {}
        losses['loss_delta_hand'] = (delta_loss_hand.sum() / hand_cal_idx.sum().item()) / 21
        return losses

    def loss_obj_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_obj_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_objkeys = outputs['pred_obj_keypoints'][idx]
        # tgt_indices = self.merge_indices(indices, outputs['pred_logits'].shape[1])
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()

        # target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # hand_cal_idx = ((target_labels == 9).to(torch.int64) + (target_labels==10).to(torch.int64)) == 1
        # obj_cal_idx = ((target_labels == 9).to(torch.int64) + (target_labels==10).to(torch.int64)) == 0

        target_6Dpose = torch.cat([t['obj6Dpose'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()
        # target_6Dpose_flipcheck = torch.cat([t['obj6Dpose'][i] * (t['flip_prob'] > 0.5) for t, (_, i) in zip(targets, indices)], dim=0)
        # FLIP 되지 않은것들
        obj_cal_idx = target_6Dpose.sum(dim=1) != 0
        # obj_cal_idx = target_6Dpose_flipcheck.sum(dim=1) != 0

        loss_objkey = F.l1_loss(src_objkeys[obj_cal_idx], target_keypoints[obj_cal_idx].view(-1, 63), reduction='none')

        losses = {}
        losses['loss_obj_keypoint'] = (loss_objkey.sum() / obj_cal_idx.sum().item()) / 21
        # losses['loss_obj_keypoint'] = torch.mean(torch.sum((target_keypoints[obj_cal_idx].view(-1, 63) - src_objkeys[obj_cal_idx])**2, dim=-1))

        return losses

    def loss_delta_obj(self, outputs, targets, indices, num_boxes):
        assert 'pred_obj_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        target_6Dpose = torch.cat([t['obj6Dpose'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()
        obj_cal_idx = target_6Dpose.sum(dim=1) != 0
        src_objkeys = outputs['pred_obj_keypoints'][idx][obj_cal_idx]


        paths = []
        for t, (_, i) in zip(targets, indices):
            paths.extend([join(*t['img_path'].split("/")[:-1])]*t['obj6Dpose'][i].shape[0])
        paths = np.array(paths)[obj_cal_idx.detach().cpu()].tolist()
        #paths = np.array([[join(*t['img_path'].split("/")[:-1])]*t['obj6Dpose'][i].shape[0] for t, (_, i) in zip(targets, indices)]).flatten()[obj_cal_idx.detach().cpu()].tolist()

        delta_loss_pred = []
        for i in range(src_objkeys.shape[0] - 1, 0, -1):
            for j in range(i):
                #if paths[i] == paths[j]:
                delta_loss_pred.append(torch.abs(src_objkeys[i] - src_objkeys[j]))

        delta_loss_pred = torch.stack(delta_loss_pred)

        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()
        target_keypoints = target_keypoints[obj_cal_idx].view(-1, 63)
        delta_loss_target = []
        for i in range(target_keypoints.shape[0] - 1, 0, -1):
            for j in range(i):
               # if paths[i] == paths[j]:
                delta_loss_target.append(torch.abs(target_keypoints[i] - target_keypoints[j]))
        delta_loss_target = torch.stack(delta_loss_target)

        delta_loss_obj = F.l1_loss(delta_loss_pred, delta_loss_target,
                                    reduction='none')

        losses = {}
        losses['loss_delta_obj'] = (delta_loss_obj.sum() / obj_cal_idx.sum().item()) / 21
        return losses


    # def perform_loss(self, output_proposals, targets, indices, num_classes, type):
    #     proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]
    #     proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()  # b, c, hw
    #
    #     target_classes = num_classes * torch.ones([proposal_cls_logits.shape[0],
    #                                                     proposal_size[0] * proposal_size[1]],
    #                                                    device=proposal_cls_logits.device)
    #     target_classes = target_classes.to(torch.int64)
    #     # left_hand_idx = (target_classes_o == 9).to(torch.bool).cuda()
    #     # right_hand_idx = (target_classes_o == 10).to(torch.bool).cuda()
    #     idx = self._get_src_permutation_idx(indices)
    #     if type == 'left':
    #         target_classes_o = torch.cat([torch.tensor([1]) for t, (_, J) in zip(targets, indices) if 9 in t['labels']]).cuda()
    #         target_classes[idx] = target_classes_o
    #     elif type == 'right':
    #         target_classes_o = torch.cat([torch.tensor([1]) for t, (_, J) in zip(targets, indices) if 10 in t['labels']]).cuda()
    #         target_classes[idx] = target_classes_o
    #     else:
    #         target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)]).cuda()
    #         target_classes[idx] = target_classes_o
    #
    #     loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
    #     return loss_proposal
    def loss_proposals(self, output_proposals, targets, indices):
        assert "proposal_cls_logits" in output_proposals
        left_proposals, right_proposals, obj_proposals = output_proposals['proposal_cls_logits']
        proposal_size = left_proposals.shape[-2:]

        target_classes_left = 1 * torch.ones([left_proposals.shape[0],
                                                           proposal_size[0] * proposal_size[1]],
                                                          device=left_proposals.device)
        target_classes_left = target_classes_left.to(torch.int64)
        target_classes_right = 1 * torch.ones([left_proposals.shape[0],
                                              proposal_size[0] * proposal_size[1]],
                                             device=left_proposals.device)
        target_classes_right = target_classes_right.to(torch.int64)
        target_classes_obj = 9 * torch.ones([left_proposals.shape[0],
                                              proposal_size[0] * proposal_size[1]],
                                             device=left_proposals.device)
        target_classes_obj = target_classes_obj.to(torch.int64)
        # left_hand_idx = (target_classes_o == 9).to(torch.bool).cuda()
        # right_hand_idx = (target_classes_o == 10).to(torch.bool).cuda()
        idx = [self._get_src_permutation_idx(indices[0]), self._get_src_permutation_idx(indices[1]), self._get_src_permutation_idx(indices[2])]
        target_classes_o_left = torch.cat([torch.tensor([0] * (len(idx[0][0]) // len(targets))) if 9 in t['labels'] else torch.tensor([-1] * (len(idx[0][0]) // len(targets))) for t, (_, J) in zip(targets, indices[0])]).cuda()
        target_classes_left[idx[0]] = target_classes_o_left
        target_classes_o_right = torch.cat([torch.tensor([0] * (len(idx[1][0]) // len(targets))) if 10 in t['labels'] else torch.tensor([-1] * (len(idx[1][0]) // len(targets))) for t, (_, J) in zip(targets, indices[1])]).cuda()
        target_classes_right[idx[1]] = target_classes_o_right
        target_classes_o_obj = torch.cat([t['labels'][-1][None].repeat(len(idx[2][0]) // len(targets)) for t, (_, J) in zip(targets, indices[2])]).cuda()
        target_classes_obj[idx[2]] = target_classes_o_obj

        loss_left = F.cross_entropy(left_proposals.flatten(2).float(), target_classes_left, ignore_index=-1)
        loss_right = F.cross_entropy(right_proposals.flatten(2).float(), target_classes_right, ignore_index=-1)
        loss_obj = F.cross_entropy(obj_proposals.flatten(2).float(), target_classes_obj, ignore_index=-1)



        # loss_left = self.perform_loss({"proposal_cls_logits": left_proposals}, targets, indices[0], 2, "left")
        # loss_right = self.perform_loss({"proposal_cls_logits": right_proposals}, targets, indices[1], 2, "right")
        # loss_obj = self.perform_loss({"proposal_cls_logits": obj_proposals}, targets, indices[2], 9, "obj")
        # proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]
        # proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()  # b, c, hw
        #
        # target_classes = self.num_classes * torch.ones([proposal_cls_logits.shape[0],
        #                                                 proposal_size[0] * proposal_size[1]],
        #                                                device=proposal_cls_logits.device)
        # target_classes = target_classes.to(torch.int64)
        #
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).cuda()
        # idx = self._get_src_permutation_idx(indices)
        # target_classes[idx] = target_classes_o
        #
        # loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
        loss_proposal = loss_left + loss_right + loss_obj
        losses = {"loss_proposal": loss_proposal}

        return losses

    def loss_contact_map(self, output_proposals, targets):
        assert "contact_map" in output_proposals
        contact_map = output_proposals["contact_map"].sigmoid()
        target_contact_map = torch.stack([t["contactmap"] for t in targets]).cuda()
        # convert to binary
        # target_contact_map = (target_contact_map > 0.4).float()
        loss_contact_map = F.l1_loss(contact_map, target_contact_map)
        losses = {"loss_contact_map": loss_contact_map}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'hand_keypoint': self.loss_hand_keypoints,
            'hand_2dkeypoint': self.loss_hand_2dkeypoints,
            'obj_keypoint': self.loss_obj_keypoints,
            'delta_hand': self.loss_delta_hand,
            'delta_obj': self.loss_delta_obj
        }

        if loss == 'delta_hand' or loss == 'delta_obj':
                #or loss == 'hand_2dkeypoint':
            # if not self.training:
            return {loss: torch.tensor(0.0).cuda()}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        proposal_loss_dict = {}
        if outputs.get("proposal_cls_logits") is not None:
            output_proposals = {"proposal_cls_logits": outputs.pop("proposal_cls_logits")}
            indices = self.matcher(output_proposals, targets)
            proposal_loss_dict = self.loss_proposals(output_proposals, targets, indices)


        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        contactmap_loss_dict = {}
        if outputs_without_aux.get("contact_map") is not None:
            output_proposals = {"contact_map": outputs_without_aux.pop("contact_map")}

            contactmap_loss_dict = self.loss_contact_map(output_proposals, targets) # {"loss_contact_map": torch.tensor(0.0).cuda()} #

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
        # proposal_loss_dict = self.loss_proposals(outputs, targets, indices)
        losses.update(proposal_loss_dict)
        losses.update(contactmap_loss_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        #
        # if 'enc_outputs' in outputs:
        #     enc_outputs = outputs['enc_outputs']
        #     bin_targets = copy.deepcopy(targets)
        #     for bt in bin_targets:
        #         bt['labels'] = torch.zeros_like(bt['labels'])
        #     indices = self.matcher(enc_outputs, bin_targets)
        #     for loss in self.losses:
        #         kwargs = {}
        #         if loss == 'labels':
        #             # Logging is enabled only for the last layer
        #             kwargs['log'] = False
        #         l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
        #         l_dict = {k + f'_enc': v for k, v in l_dict.items()}
        #         losses.update(l_dict)

        return losses