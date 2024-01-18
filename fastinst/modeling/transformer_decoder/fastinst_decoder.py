import math

import torch
from detectron2.config import configurable
from torch import nn
from torch.nn import functional as F

from fastinst.utils.misc import nested_tensor_from_tensor_list
from .utils import TRANSFORMER_DECODER_REGISTRY, QueryProposal, \
    CrossAttentionLayer, SelfAttentionLayer, FFNLayer, MLP
from ... import GraFormer

@TRANSFORMER_DECODER_REGISTRY.register()
class FastInstDecoder(nn.Module):

    @configurable
    def __init__(
            self,
            in_channels,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            num_aux_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            num_contact_queries: int,

    ):
        """
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            num_aux_queries: number of auxiliary queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
        """
        super().__init__()
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.iam_queries = num_queries
        self.contact_map_queries = num_contact_queries
        self.num_queries = self.iam_queries + self.contact_map_queries
        self.num_aux_queries = num_aux_queries
        self.criterion = None

        meta_pos_size = int(round(math.sqrt(self.iam_queries)))
        self.meta_pos_embed = nn.Parameter(torch.empty(1, hidden_dim, meta_pos_size, meta_pos_size))
        if num_aux_queries > 0:
            self.empty_query_features = nn.Embedding(num_aux_queries, hidden_dim)
            self.empty_query_pos_embed = nn.Embedding(num_aux_queries, hidden_dim)


        self.query_proposal = QueryProposal(hidden_dim, num_queries, num_classes, num_contact_queries)

        self.transformer_query_cross_attention_layers = nn.ModuleList()
        self.transformer_query_self_attention_layers = nn.ModuleList()
        self.transformer_query_ffn_layers = nn.ModuleList()
        self.transformer_mask_cross_attention_layers = nn.ModuleList()
        self.transformer_mask_ffn_layers = nn.ModuleList()

        for idx in range(self.num_layers):
            self.transformer_query_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )

        self.decoder_query_norm_layers = nn.ModuleList()
        self.class_embed_layers = nn.ModuleList()
        self.mask_embed_layers = nn.ModuleList()
        self.mask_features_layers = nn.ModuleList()
        self.keypoint_embed_left = MLP(hidden_dim, hidden_dim, 42, 3)
        self.keypoint_embed_right = MLP(hidden_dim, hidden_dim, 42, 3)
        self.obj_keypoint_embed = MLP(hidden_dim, hidden_dim, 63, 3)
        self.class_embed_layers = nn.Linear(hidden_dim, num_classes)

        # nn.init.constant_(self.keypoint_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.keypoint_embed.layers[-1].bias.data, 0)

        nn.init.constant_(self.keypoint_embed_left.layers[-1].weight.data, 0)
        nn.init.constant_(self.keypoint_embed_left.layers[-1].bias.data, 0)
        nn.init.constant_(self.keypoint_embed_right.layers[-1].weight.data, 0)
        nn.init.constant_(self.keypoint_embed_right.layers[-1].bias.data, 0)

        nn.init.constant_(self.obj_keypoint_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_keypoint_embed.layers[-1].bias.data, 0)
        for idx in range(self.num_layers + 1):
            self.decoder_query_norm_layers.append(nn.LayerNorm(hidden_dim))
            # self.mask_embed_layers.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))
            # self.mask_features_layers.append(nn.Linear(hidden_dim, mask_dim))
        self.keypoint_embed_left = nn.ModuleList([self.keypoint_embed_left for _ in range(self.num_layers+1)])
        self.keypoint_embed_right = nn.ModuleList([self.keypoint_embed_right for _ in range(self.num_layers+1)])
        self.obj_keypoint_embed = nn.ModuleList([self.obj_keypoint_embed for _ in range(self.num_layers+1)])
        self.class_embed_layers = nn.ModuleList([self.class_embed_layers for _ in range(self.num_layers+1)])
        edges = GraFormer.create_edges(num_nodes=21)
        adj = GraFormer.adj_mx_from_edges(num_pts=21, edges=edges, sparse=False)
        self.graformer_left = GraFormer.GraFormer(adj=adj.cuda(), hid_dim=64, coords_dim=(2, 3),
                                        n_pts=21, num_layers=2, n_head=4, dropout=0.1)
        self.graformer_right = GraFormer.GraFormer(adj=adj.cuda(), hid_dim=64, coords_dim=(2, 3),
                                                  n_pts=21, num_layers=2, n_head=4, dropout=0.1)
        self.graformer_left = nn.ModuleList([self.graformer_left for _ in range(self.num_layers+1)])
        self.graformer_right = nn.ModuleList([self.graformer_right for _ in range(self.num_layers+1)])
        # self.contact_embeded = nn.Linear(778, 256)



    @classmethod
    def from_config(cls, cfg, in_channels, input_shape):
        ret = {}
        ret["in_channels"] = in_channels

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.FASTINST.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.FASTINST.NUM_OBJECT_QUERIES
        ret["num_aux_queries"] = cfg.MODEL.FASTINST.NUM_AUX_QUERIES
        ret["num_contact_queries"] = cfg.MODEL.FASTINST.NUM_CONTACT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.FASTINST.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.FASTINST.DIM_FEEDFORWARD

        ret["dec_layers"] = cfg.MODEL.FASTINST.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.FASTINST.PRE_NORM

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, targets=None):
        bs = x[0].shape[0]
        proposal_size = x[1].shape[-2:]
        pixel_feature_size = x[2].shape[-2:]

        pixel_pos_embeds = F.interpolate(self.meta_pos_embed, size=pixel_feature_size,
                                         mode="bilinear", align_corners=False)
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=proposal_size,
                                            mode="bilinear", align_corners=False)

        pixel_features = x[2].flatten(2).permute(2, 0, 1)
        pixel_pos_embeds = pixel_pos_embeds.flatten(2).permute(2, 0, 1)

        query_features, query_pos_embeds, query_locations, proposal_cls_logits, contact_map = self.query_proposal(
            x[1], proposal_pos_embeds, targets
        )

        # contact_map_queries = contact_map # .clone()
        # # repeat contact map for each query
        # contact_map_queries = contact_map_queries.view(contact_map_queries.shape[0] * contact_map_queries.shape[1], -1)
        # contact_map_queries = self.contact_embeded(contact_map_queries)
        # contact_map_queries = contact_map_queries.unsqueeze(0).repeat(self.num_queries, 1, 1)

        # contact_map_queries = contact_map_queries.permute(2, 0, 1)
        # contact_map_pos_embeddings = contact_map_pos_embeddings.permute(2, 0, 1)
        query_features = query_features.permute(2, 0, 1)
        query_pos_embeds = query_pos_embeds.permute(2, 0, 1)

        # query_features = torch.cat([query_features, contact_map_queries])
        # query_pos_embeds = torch.cat([query_pos_embeds, contact_map_pos_embeddings])
        # temporary contact query features
        # contact_map_queries_left = contact_map_queries[:, 0:2:contact_map_queries.shape[1], :]
        # contact_map_queries_right = contact_map_queries[:, 1:2:contact_map_queries.shape[1], :]
        # contact_map_queries = torch.stack([contact_map_queries_left, contact_map_queries_right])
        # query_features = query_features + contact_map_queries

        if self.num_aux_queries > 0:
            aux_query_features = self.empty_query_features.weight.unsqueeze(1).repeat(1, bs, 1)
            aux_query_pos_embed = self.empty_query_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_features = torch.cat([query_features, aux_query_features], dim=0)
            query_pos_embeds = torch.cat([query_pos_embeds, aux_query_pos_embed], dim=0)

        outputs_class, outputs_mask, attn_mask, _, _, key_preds = self.forward_prediction_heads(
            query_features, pixel_features, pixel_feature_size, -1, return_attn_mask=False
        )
        predictions_class = [outputs_class]
        predictions_mask = [outputs_mask]
        predictions_matching_index = [None]
        query_feature_memory = [query_features]
        pixel_feature_memory = [pixel_features]
        key, obj_key = key_preds
        outputs_keypoints = [key.sigmoid()]
        outputs_obj_keypoints = [obj_key.sigmoid()]

        for i in range(self.num_layers):
            query_features, pixel_features = self.forward_one_layer(
                query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i
            )
            if i < self.num_layers - 1:
                outputs_class, outputs_mask, attn_mask, _, _, key_preds = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, i, return_attn_mask=False,
                )
            else:
                outputs_class, outputs_mask, _, matching_indices, gt_attn_mask, key_preds = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, i,
                    return_gt_attn_mask=self.training, targets=targets, query_locations=query_locations
                )
            key, obj_key = key_preds

            outputs_keypoint = key.sigmoid()  # modify
            outputs_obj_keypoint = obj_key.sigmoid()  # modify
            outputs_keypoints.append(outputs_keypoint)  # modify
            outputs_obj_keypoints.append(outputs_obj_keypoint)  # modify

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_matching_index.append(None)
            query_feature_memory.append(query_features)
            pixel_feature_memory.append(pixel_features)

        guided_predictions_class = []
        guided_obj_keypoints = []
        guided_keypoints = []
        if self.training:
            for i in range(self.num_layers):
                query_features, pixel_features = self.forward_one_layer(
                    query_feature_memory[i + 1], pixel_feature_memory[i + 1], query_pos_embeds,
                    pixel_pos_embeds, gt_attn_mask, i
                )

                outputs_class, outputs_mask, _, _, _, key_preds = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, idx_layer=i, return_attn_mask=False
                )
                guided_keypoints.append(key_preds[0].sigmoid())
                guided_obj_keypoints.append(key_preds[1].sigmoid())
                guided_predictions_class.append(outputs_class)
        predictions_class = guided_predictions_class + predictions_class
        outputs_keypoints = torch.stack(guided_keypoints + outputs_keypoints)  # modify
        outputs_obj_keypoints = torch.stack(guided_obj_keypoints + outputs_obj_keypoints)  # modify

        # predictions_mask = guided_predictions_mask + predictions_mask
        # predictions_matching_index = guided_predictions_matching_index + predictions_matching_index
        out = {
            'proposal_cls_logits': proposal_cls_logits,
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_matching_indices': predictions_matching_index[-1],
            'pred_keypoints': outputs_keypoints[-1], 'pred_obj_keypoints': outputs_obj_keypoints[-1],
            'query_locations': query_locations if query_locations is not None else query_locations,
            "contact_map": contact_map,
        }
        if self.training:
            out['aux_outputs'] = self._set_aux_loss(
                predictions_class, predictions_mask, predictions_matching_index, query_locations, outputs_keypoints, outputs_obj_keypoints)
        return out

    def forward_one_layer(self, query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i):
        pixel_features = self.transformer_mask_cross_attention_layers[i](
            pixel_features, query_features, query_pos=pixel_pos_embeds, pos=query_pos_embeds
        )
        pixel_features = self.transformer_mask_ffn_layers[i](pixel_features)

        query_features = self.transformer_query_cross_attention_layers[i](
            query_features, pixel_features, memory_mask=attn_mask, query_pos=query_pos_embeds, pos=pixel_pos_embeds
        )
        query_features = self.transformer_query_self_attention_layers[i](
            query_features, query_pos=query_pos_embeds
        )
        query_features = self.transformer_query_ffn_layers[i](query_features)
        return query_features, pixel_features

    def forward_prediction_heads(self, query_features, pixel_features, pixel_feature_size, idx_layer, contact_map=None,
                                 return_attn_mask=False, return_gt_attn_mask=False,
                                 targets=None, query_locations=None):
        decoder_query_features = self.decoder_query_norm_layers[idx_layer + 1](query_features[:self.num_queries])
        decoder_query_features = decoder_query_features.transpose(0, 1)
        if self.training or idx_layer + 1 == self.num_layers:
            outputs_class = self.class_embed_layers[idx_layer + 1](decoder_query_features)
        else:
            outputs_class = None
        # outputs_mask_embed = self.mask_embed_layers[idx_layer + 1](decoder_query_features)
        # outputs_mask_features = self.mask_features_layers[idx_layer + 1](pixel_features.transpose(0, 1))

        # outputs_mask = torch.einsum("bqc,blc->bql", outputs_mask_embed, outputs_mask_features)
        # outputs_mask = outputs_mask.reshape(-1, self.num_queries, *pixel_feature_size)
        obj_key = self.obj_keypoint_embed[idx_layer + 1](decoder_query_features)  # modify
        key_left = self.keypoint_embed_left[idx_layer + 1](decoder_query_features) # + contact_map[0].permute(1, 0, 2))  # modify
        key_right = self.keypoint_embed_right[idx_layer + 1](decoder_query_features) # + contact_map[1].permute(1, 0, 2))  # modify
        key_left_graformer = self.graformer_left[idx_layer + 1](key_left.view(key_left.shape[0] * key_left.shape[1], 21, 2)[..., :2])
        key_right_graformer = self.graformer_right[idx_layer + 1](key_right.view(key_right.shape[0] * key_right.shape[1], 21, 2)[..., :2])
        key_left = key_left.view(key_left.shape[0], key_left.shape[1], 21, 2)
        key_right = key_right.view(key_right.shape[0], key_right.shape[1], 21, 2)
        key_left = torch.concat((key_left, torch.ones((key_left.shape[0], key_left.shape[1], 21, 1)).cuda()), dim=3)
        key_right = torch.concat((key_right, torch.ones((key_right.shape[0], key_right.shape[1], 21, 1)).cuda()), dim=3)
        key = torch.stack([
                           key_left_graformer.view(key_left.shape[0], key_left.shape[1], 63),
                           key_right_graformer.view(key_right.shape[0], key_right.shape[1], 63),
                           key_left.view(key_left.shape[0], key_left.shape[1], 63), key_right.view(key_right.shape[0], key_right.shape[1], 63)])
        # key = key.reshape(key.shape[0], key.shape[1], 21, 3)
        # key[..., :2] += reference[:, :self.num_queries, None, :]  # modify
        # key = key.reshape(key.shape[0], key.shape[1], -1)
        #
        # obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], 21, 3)
        # obj_key[..., :2] += reference[:, :self.num_queries, None, :]  # modify
        # obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], -1)
        if return_attn_mask:
            # outputs_mask.shape: b, q, h, w
            attn_mask = F.pad(outputs_mask, (0, 0, 0, 0, 0, self.num_aux_queries), "constant", 1)
            attn_mask = (attn_mask < 0.).flatten(2)  # b, q, hw
            invalid_query = attn_mask.all(-1, keepdim=True)  # b, q, 1
            attn_mask = (~ invalid_query) & attn_mask  # b, q, hw
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None

        if return_gt_attn_mask:
            assert targets is not None and query_locations is not None
            matching_indices = self.criterion.matcher(
                {'pred_logits': outputs_class,
                 'query_locations': query_locations, 'pred_keypoints': key.sigmoid(), 'pred_obj_keypoints': obj_key.sigmoid()}, targets)
            # matching_indices = self.criterion.merge_indices(matching_indices, outputs_class.shape[1])
            src_idx = self.criterion._get_src_permutation_idx(matching_indices)
            tgt_idx = self.criterion._get_tgt_permutation_idx(matching_indices)

            mask = [t["masks"] for t in targets]
            target_mask, valid = nested_tensor_from_tensor_list(mask).decompose()
            if target_mask.shape[1] > 0:
                target_mask = target_mask.to(key)
                target_mask = F.interpolate(target_mask, size=pixel_feature_size, mode="nearest").bool()
            else:
                target_mask_size = [target_mask.shape[0], target_mask.shape[1], *pixel_feature_size]
                target_mask = torch.zeros(size=target_mask_size, device=key.device).bool()

            gt_attn_mask_size = [
                target_mask.shape[0], self.num_queries + self.num_aux_queries, *pixel_feature_size
            ]
            gt_attn_mask = torch.zeros(size=gt_attn_mask_size, device=key.device).bool()
            gt_attn_mask[src_idx] = ~ target_mask[tgt_idx]
            gt_attn_mask = gt_attn_mask.flatten(2)

            invalid_gt_query = gt_attn_mask.all(-1, keepdim=True)  # b, n, 1
            gt_attn_mask = (~invalid_gt_query) & gt_attn_mask
            gt_attn_mask = gt_attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            gt_attn_mask = gt_attn_mask.detach()
        else:
            matching_indices = None
            gt_attn_mask = None

        return outputs_class, None, attn_mask, matching_indices, gt_attn_mask, (key, obj_key)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, output_indices, output_query_locations, output_keyp, output_obj_keyp):
        return [
            {
                "query_locations": output_query_locations,
                "pred_logits": a,
                "pred_masks": b,
                "pred_matching_indices": c,
                "pred_keypoints": d,
                "pred_obj_keypoints": e}
            for a, b, c, d, e in zip(outputs_class[:-1], outputs_seg_masks[:-1], output_indices[:-1], output_keyp[:-1], output_obj_keyp[:-1])
        ]
