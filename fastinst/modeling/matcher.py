# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import point_sample
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast


#
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


#
#
batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def class_cost(out_prob, tgt_ids):
    # Compute the classification cost.
    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
    return cost_class


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


#
#
batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


# class HungarianMatcher(nn.Module):
#     """This class computes an assignment between the targets and the predictions of the network
#
#     For efficiency reasons, the targets don't include the no_object. Because of this, in general,
#     there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
#     while the others are un-matched (and thus treated as non-objects).
#     """
#
#     def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, cost_location: float = 1e3,
#                  num_points: int = 0):
#         """Creates the matcher
#
#         Params:
#             cost_class: This is the relative weight of the classification error in the matching cost
#             cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
#             cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
#             cost_location: This is the relative weight of the location loss of the query in the matching cost
#         """
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_mask = cost_mask
#         self.cost_dice = cost_dice
#         self.cost_location = cost_location
#
#         assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
#
#         self.num_points = num_points
#

#
#     @torch.no_grad()
#     def memory_efficient_forward(self, outputs, targets):
#         """memory-friendly matching"""
#         bs, num_queries = outputs["pred_logits"].shape[:2]
#
#         indices = []
#         # Iterate through batch size
#         for b in range(bs):
#             out_query_loc = outputs["query_locations"][b]  # [num_queries, 2(x, y)]
#             out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
#             out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
#             # gt masks are already padded when preparing target
#             tgt_mask = targets[b]["masks"].to(out_mask)  # [num_obj, h, w]
#             tgt_ids = targets[b]["labels"]
#
#             cost_location = point_sample(
#                 tgt_mask.unsqueeze(0),
#                 out_query_loc.unsqueeze(0),
#                 align_corners=False
#             ).squeeze(0)  # [num_obj, num_queries]
#             cost_location = (cost_location > 0).to(out_mask)
#             # add location cost when the proposal is not inside instance regions.
#             cost_location = -cost_location.transpose(0, 1)  # [num_queries, num_obj]
#
#             # Compute the classification cost. Contrary to the loss, we don't use the NLL,
#             # but approximate it in 1 - proba[target class].
#             # The 1 is a constant that doesn't change the matching, it can be ommitted.
#             cost_class = -out_prob[:, tgt_ids]  # [num_queries, num_obj]
#
#             # all masks share the same set of points for efficient matching!
#             point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
#             # get gt labels
#             tgt_mask = point_sample(
#                 tgt_mask.unsqueeze(0),
#                 point_coords,
#                 align_corners=False,
#             ).squeeze(0)
#
#             out_mask = point_sample(
#                 out_mask.unsqueeze(0),
#                 point_coords,
#                 align_corners=False,
#             ).squeeze(0)
#
#             with autocast(enabled=False):
#                 out_mask = out_mask.float()
#                 tgt_mask = tgt_mask.float()
#                 # Compute the focal loss between masks
#                 cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
#
#                 # Compute the dice loss between masks
#                 if tgt_mask.shape[0] > 0:
#                     cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
#                 else:
#                     cost_dice = batch_dice_loss(out_mask, tgt_mask)
#
#             # Final cost matrix
#             C = (
#                     self.cost_mask * cost_mask
#                     + self.cost_class * cost_class
#                     + self.cost_dice * cost_dice
#                     + self.cost_location * cost_location
#             )
#             C = C.reshape(num_queries, -1).cpu()
#             indices.append(linear_sum_assignment(C))
#
#         return [
#             (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#             for i, j in indices
#         ]
#
#     @torch.no_grad()
#     def memory_efficient_forward_for_proposal(self, outputs, targets):
#         """memory-friendly matching for proposals"""
#         bs = outputs["proposal_cls_logits"].shape[0]
#         proposal_size = outputs["proposal_cls_logits"].shape[-2:]
#
#         indices = []
#         # Iterate through batch size
#         for b in range(bs):
#             proposal_cls_prob = outputs["proposal_cls_logits"][b].flatten(1).transpose(0, 1).softmax(
#                 -1)  # [proposal_hw, num_classes]
#
#             # gt masks are already padded when preparing target
#             tgt_mask = targets[b]["masks"].to(proposal_cls_prob)  # [num_obj, h, w]
#             tgt_ids = targets[b]["labels"]
#
#             if tgt_mask.shape[0] > 0:
#                 scaled_tgt_mask = F.adaptive_avg_pool2d(tgt_mask.unsqueeze(0),
#                                                         output_size=proposal_size)
#                 scaled_tgt_mask = (scaled_tgt_mask.squeeze(0) > 0.).to(
#                     proposal_cls_prob)  # [num_obj, proposal_h ,proposal_w]
#             else:
#                 scaled_tgt_mask = torch.zeros([tgt_mask.shape[0], *proposal_size],
#                                               device=proposal_cls_prob.device)
#
#             # add location cost when the proposal is not inside the instance region.
#             cost_location = -scaled_tgt_mask.flatten(1).transpose(0, 1)  # [proposal_hw, num_obj]
#
#             # Compute the classification cost. Contrary to the loss, we don't use the NLL,
#             # but approximate it in 1 - proba[target class].
#             # The 1 is a constant that doesn't change the matching, it can be omitted.
#             cost_class = -proposal_cls_prob[:, tgt_ids]  # [proposal_hw, num_obj]
#
#             # Proposal cost matrix
#             C = self.cost_class * cost_class + self.cost_location * cost_location
#             C = C.reshape(proposal_size[0] * proposal_size[1], -1).cpu()
#             indices.append(linear_sum_assignment(C))
#
#         return [
#             (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#             for i, j in indices
#         ]
#
#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         """Performs the matching
#
#         Params:
#             outputs: This is a dict that contains at least these entries:
#                  "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
#                  "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks
#
#             targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
#                  "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
#                            objects in the target) containing the class labels
#                  "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
#
#         Returns:
#             A list of size batch_size, containing tuples of (index_i, index_j) where:
#                 - index_i is the indices of the selected predictions (in order)
#                 - index_j is the indices of the corresponding selected targets (in order)
#             For each batch element, it holds:
#                 len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
#         """
#         if outputs.get("proposal_cls_logits", None) is not None:
#             return self.memory_efficient_forward_for_proposal(outputs, targets)
#         return self.memory_efficient_forward(outputs, targets)
#
#     def __repr__(self, _repr_indent=4):
#         head = "Matcher " + self.__class__.__name__
#         body = [
#             "cost_class: {}".format(self.cost_class),
#             "cost_mask: {}".format(self.cost_mask),
#             "cost_dice: {}".format(self.cost_dice),
#         ]
#         lines = [head] + [" " * _repr_indent + line for line in body]
#         return "\n".join(lines)

# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, cost_location: float = 1e3,
                                   num_points: int = 0,
                 cost_keypoint: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_keypoint: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_keypoint = cost_keypoint
        self.cost_location = cost_location
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        assert cost_class != 0 or cost_keypoint != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        if outputs.get("proposal_cls_logits", None) is not None:
            return self.memory_efficient_forward_for_proposal(outputs, targets)
        return self.forward_main(outputs, targets)

    def forward_main(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            # out_prob_hand = outputs["pred_logits"][:, :num_queries // 2].flatten(0, 1).sigmoid()
            # out_prob_obj = outputs["pred_logits"][:, num_queries // 2:].flatten(0, 1).sigmoid()
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            # out_kp = outputs["pred_keypoints"].flatten(0, 1)
            key_left, key_right, _, _ = outputs["pred_keypoints"]
            key_left = key_left.flatten(0, 1)
            key_right = key_right.flatten(0, 1)
            out_objkp = outputs["pred_obj_keypoints"].flatten(0, 1)

            # Also concat the target labels and boxes
            try:
                tgt_ids = torch.cat([v["labels"] for v in targets]).cuda()
            except:
                print("error")
            tgt_kp = torch.cat([v["keypoints"] for v in targets]).cuda()

            # total_out_kp = out_kp.clone()
            # obj_idx = (out_prob.argmax(dim=1) != 9) *  (out_prob.argmax(dim=1) != 10)
            # total_out_kp[obj_idx] = out_objkp[obj_idx].clone()
            # object에 해당하는 index에는 obj_keyembedd값을 넣음.
            left_hand_idx = (tgt_ids == 9)
            right_hand_idx = (tgt_ids == 10)
            hand_idx = left_hand_idx + right_hand_idx
            obj_idx = (hand_idx == 0)
            # class_hand = class_cost(out_prob_hand, tgt_ids[hand_idx])
            # class_obj = class_cost(out_prob_obj, tgt_ids[obj_idx])
            cost_class = class_cost(out_prob, tgt_ids)
            # Compute the L1 cost between boxes
            # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            # cost_bbox = torch.cdist(total_out_kp.reshape(-1,21,3)[...,:2].reshape(-1,42), tgt_kp[...,:2].reshape(-1, 42), p=1) / 21
            costs = []
            if out_objkp.sum() > 0:
                cost_keypoints = torch.zeros_like(cost_class)
                # cost_hand = torch.cdist(out_kp, tgt_kp.reshape(-1, 63)[hand_idx], p=1)
                cost_hand_left = torch.cdist(key_left, tgt_kp.reshape(-1, 63)[left_hand_idx], p=1)
                cost_hand_right = torch.cdist(key_right, tgt_kp.reshape(-1, 63)[right_hand_idx], p=1)
                cost_obj = torch.cdist(out_objkp, tgt_kp.reshape(-1, 63)[obj_idx], p=1)
                for b in range(bs):
                    out_query_loc = outputs["query_locations"][b]  # [num_queries, 2(x, y)]
                    tgt_mask = targets[b]["masks"].to(out_query_loc)  # [num_obj, h, w]
                    cost_location = point_sample(
                        tgt_mask.unsqueeze(0),
                        out_query_loc.unsqueeze(0),
                        align_corners=False
                    ).squeeze(0)  # [num_obj, num_queries]
                    cost_location = (cost_location > 0).to(out_query_loc)
                    cost_location = -cost_location.transpose(0, 1)
                    costs.append(cost_location)

                    # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                    # # get gt labels
                    # tgt_mask = point_sample(
                    #     tgt_mask.unsqueeze(0),
                    #     point_coords,
                    #     align_corners=False,
                    # ).squeeze(0)
                    #
                    # out_mask = point_sample(
                    #     out_mask.unsqueeze(0),
                    #     point_coords,
                    #     align_corners=False,
                    # ).squeeze(0)
                    #
                    # with autocast(enabled=False):
                    #     out_mask = out_mask.float()
                    #     tgt_mask = tgt_mask.float()
                    #     # Compute the focal loss between masks
                    #     cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                    #
                    #     # Compute the dice loss between masks
                    #     if tgt_mask.shape[0] > 0:
                    #         cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
                    #     else:
                    #         cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    # total_cost = cost_dice + cost_mask + cost_location
                    # costs.append(total_cost)

                # cost_keypoints[:, hand_idx] = cost_hand
                cost_keypoints[:, left_hand_idx] = cost_hand_left
                cost_keypoints[:, right_hand_idx] = cost_hand_right
                cost_keypoints[:, obj_idx] = cost_obj
                # Compute the giou cost betwen boxes
                # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                #                                  box_cxcywh_to_xyxy(tgt_bbox))
                # cost_location = torch.stack(costs).view(num_queries, -1).repeat(8, 1)
                # C_hand = self.cost_keypoint * cost_hand + self.cost_class * class_hand #+ self.cost_location * cost_location
                # C_obj = self.cost_keypoint * cost_obj + self.cost_class * class_obj #+ self.cost_location * cost_location
                C = self.cost_keypoint * cost_keypoints + self.cost_class * cost_class
                # C_hand = C_hand.view(bs, num_queries // 2, -1).cpu()
                # C_obj = C_obj.view(bs, num_queries // 2, -1).cpu()
                # C = [C_hand, C_obj]
                C = C.view(bs, num_queries, -1).cpu()
            else:
                # cost_keypoints = torch.zeros_like(class_hand)
                # cost_class = torch.zeros_like(class_obj)
                cost_keypoints = torch.zeros_like(cost_class)

                C = self.cost_keypoint * cost_keypoints + self.cost_class * cost_class  # + self.cost_giou * cost_abs_depth
                C = C.view(bs, num_queries, -1).cpu()
            # sizes_hand = [2 if len(v["keypoints"]) == 3 else 1 for v in targets]
            # sizes_obj = [1 for v in targets]
            sizes = [len(v["keypoints"]) for v in targets]
            # indices_hand = [linear_sum_assignment(c[i]) for i, c in enumerate(C[0].split(sizes_hand, -1))]
            indices = [linear_sum_assignment(c[i] + self.cost_location * costs[i].cpu()) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def matching_for_proposal(self, outputs, targets, type, number):
        bs = outputs["proposal_cls_logits"].shape[0]
        proposal_size = outputs["proposal_cls_logits"].shape[-2:]
        indices_all = []
        # Iterate through batch size
        with torch.no_grad():
            for b in range(bs):
                indices_x = []
                indices_y = []
                proposal_cls_prob = outputs["proposal_cls_logits"][b].flatten(1).transpose(0, 1).softmax(
                    -1)  # [proposal_hw, num_classes]

                # gt masks are already padded when preparing target
                tgt_mask = targets[b]["masks"].to(proposal_cls_prob)  # [num_obj, h, w]
                tgt_ids = targets[b]["labels"]
                if type == "left":
                    tgt_mask = tgt_mask[tgt_ids == 9]
                    tgt_ids = tgt_ids[tgt_ids == 9]
                elif type == 'right':
                    tgt_mask = tgt_mask[tgt_ids == 10]
                    tgt_ids = tgt_ids[tgt_ids == 10]
                else:
                    tgt_mask = tgt_mask[((tgt_ids == 9) + (tgt_ids == 10)) == False]
                    tgt_ids = tgt_ids[((tgt_ids == 9) + (tgt_ids == 10)) == False]
                if len(tgt_mask) > 0 and tgt_mask.shape[0] > 0:
                    scaled_tgt_mask = F.adaptive_avg_pool2d(tgt_mask.unsqueeze(0),
                                                            output_size=proposal_size)
                    scaled_tgt_mask = (scaled_tgt_mask.squeeze(0) > 0.).to(
                        proposal_cls_prob)  # [num_obj, proposal_h ,proposal_w]
                else:
                    scaled_tgt_mask = torch.zeros([tgt_mask.shape[0], *proposal_size],
                                                  device=proposal_cls_prob.device)

                # add location cost when the proposal is not inside the instance region.
                cost_location = -scaled_tgt_mask.flatten(1).transpose(0, 1)  # [proposal_hw, num_obj]

                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be omitted.
                if len(tgt_ids) == 0:
                    cost_class = torch.zeros_like(cost_location)
                elif type == 'left' or type == 'right':
                    cost_class = -proposal_cls_prob[:, 0][:, None]  # [proposal_hw, num_obj]
                else:
                    cost_class = -proposal_cls_prob[:, tgt_ids]
                cost_clone = cost_class.clone()
                for _ in range(number):
                    # Proposal cost matrix
                    if len(tgt_ids) == 0:
                        indices_x.append(torch.as_tensor([False]))
                        indices_y.append(torch.as_tensor([False]))
                        continue
                    C = self.cost_class * cost_clone + self.cost_location * cost_location
                    C = C.reshape(proposal_size[0] * proposal_size[1], -1).cpu()
                    assign = linear_sum_assignment(C)
                    indices_x.append(torch.as_tensor(assign[0], dtype=torch.int64))
                    indices_y.append(torch.as_tensor(assign[1], dtype=torch.int64))
                    # indices.append(linear_sum_assignment(C))
                    cost_clone[indices_x[-1]] = torch.inf
                indices_all.append((torch.cat(indices_x), torch.cat(indices_y)))
            # return [
            #     (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            #     for i, j in indices_all
            # ]
        return indices_all


    def memory_efficient_forward_for_proposal(self, outputs, targets):
        """memory-friendly matching for proposals"""
        # bs = outputs["proposal_cls_logits"].shape[0]
        # proposal_size = outputs["proposal_cls_logits"].shape[-2:]
        #
        # indices = []
        # # Iterate through batch size
        # for b in range(bs):
        #     proposal_cls_prob = outputs["proposal_cls_logits"][b].flatten(1).transpose(0, 1).softmax(
        #         -1)  # [proposal_hw, num_classes]
        #
        #     # gt masks are already padded when preparing target
        #     tgt_mask = targets[b]["masks"].to(proposal_cls_prob)  # [num_obj, h, w]
        #     tgt_ids = targets[b]["labels"]
        #
        #     if tgt_mask.shape[0] > 0:
        #         scaled_tgt_mask = F.adaptive_avg_pool2d(tgt_mask.unsqueeze(0),
        #                                                 output_size=proposal_size)
        #         scaled_tgt_mask = (scaled_tgt_mask.squeeze(0) > 0.).to(
        #             proposal_cls_prob)  # [num_obj, proposal_h ,proposal_w]
        #     else:
        #         scaled_tgt_mask = torch.zeros([tgt_mask.shape[0], *proposal_size],
        #                                       device=proposal_cls_prob.device)
        #
        #     # add location cost when the proposal is not inside the instance region.
        #     cost_location = -scaled_tgt_mask.flatten(1).transpose(0, 1)  # [proposal_hw, num_obj]
        #
        #     # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        #     # but approximate it in 1 - proba[target class].
        #     # The 1 is a constant that doesn't change the matching, it can be omitted.
        #     cost_class = -proposal_cls_prob[:, tgt_ids]  # [proposal_hw, num_obj]
        #
        #     # Proposal cost matrix
        #     C = self.cost_class * cost_class + self.cost_location * cost_location
        #     C = C.reshape(proposal_size[0] * proposal_size[1], -1).cpu()
        #     indices.append(linear_sum_assignment(C))
        with torch.no_grad():
            left_proposals, right_proposals, obj_proposals = outputs['proposal_cls_logits']
            left_indices = self.matching_for_proposal({"proposal_cls_logits": left_proposals}, targets, "left", 40)
            right_indices = self.matching_for_proposal({"proposal_cls_logits": right_proposals}, targets, "right", 40)
            obj_indices = self.matching_for_proposal({"proposal_cls_logits": obj_proposals}, targets, "obj", 20)

        return [left_indices, right_indices, obj_indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_keypoint=args.set_cost_keypoint)
