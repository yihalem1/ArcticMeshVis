# Modified from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from typing import Optional

import torch
from detectron2.utils.registry import Registry
from torch import nn, Tensor
from torch.nn import functional as F
from fastinst.utils.misc import inverse_sigmoid
import numpy as np
TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in FastInst.
"""


def build_transformer_decoder(cfg, in_channels, input_shape=None):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.FASTINST.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, input_shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(tensor, max_len=512):
    """
    Adds positional encoding to a tensor.

    Args:
    - tensor (torch.Tensor): The input tensor with shape [batch_size, sequence_length, hidden_size].
    - max_len (int): Maximum length of the sequence.

    Returns:
    - torch.Tensor: The tensor with positional encodings added.
    """
    batch_size, seq_len, hidden_size = tensor.size()

    # Calculate positional encodings
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / hidden_size))
    pos_enc = position * div_term

    # Apply sine to even indices and cosine to odd indices
    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])

    # Expand positional encodings to match the batch size and sequence length
    pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)


    return pos_enc

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class QueryProposal(nn.Module):

    def __init__(self, num_features, num_queries, num_classes, num_contact_queries, max_contct_queries_pos_embeddings = 1000):
        super().__init__()
        self.topk = num_queries
        self.num_classes = num_classes
        self.num_contact_queries = num_contact_queries
        self.count = 0

        self.conv_proposal_left_cls_logits = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 2, kernel_size=1, stride=1, padding=0),
        )

        self.conv_proposal_right_cls_logits = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 2, kernel_size=1, stride=1, padding=0),
        )

        self.conv_proposal_obj_cls_logits = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_classes - 2 + 1, kernel_size=1, stride=1, padding=0),
        )

        # self.contactmap = ContactMapCNNWithTopKEmbedding(num_features, num_features, k = self.num_contact_queries, max_positional_embeddings=max_contct_queries_pos_embeddings)
        # # testing
        # self.contactmap_features = nn.Linear(2 * 778 * 1, num_features)
        # self.contact_map_features = num_features
        # # self.positional_embedding = nn.Embedding(1000, num_features)

        self.contactmap_estimation = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((778, 1)),
            nn.Flatten(),
            nn.Linear(778 * 32, 778 * 2),
        )

        self.contactmap_conv = ContactMapConvolution(
            in_channels=2,
            out_channels=num_features,
            kernel_size=3,
            num_queries=self.topk,
            num_features=num_features,
        )
    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(0, 1, h, device=x.device)
        x_loc = torch.linspace(0, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        locations = torch.stack([x_loc, y_loc], 0).unsqueeze(0)
        return locations

    def seek_local_maximum(self, x, epsilon=1e-6):
        """
        inputs:
            x: torch.tensor, shape [b, c, h, w]
        return:
            torch.tensor, shape [b, c, h, w]
        """
        x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
        # top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
        maximum = (x >= x_pad[:, :, :-2, 1:-1]) & \
                  (x >= x_pad[:, :, 2:, 1:-1]) & \
                  (x >= x_pad[:, :, 1:-1, :-2]) & \
                  (x >= x_pad[:, :, 1:-1, 2:]) & \
                  (x >= x_pad[:, :, :-2, :-2]) & \
                  (x >= x_pad[:, :, :-2, 2:]) & \
                  (x >= x_pad[:, :, 2:, :-2]) & \
                  (x >= x_pad[:, :, 2:, 2:]) & \
                  (x >= epsilon)
        return maximum.to(x)


    def save_mask(self, topk_indices, shape, type):
            mask = np.zeros(shape)
            mask = mask.flatten()
            mask[topk_indices.cpu()] = 255
            import cv2
            self.count += 1
            cv2.imwrite(f"queries/mask_{type}_{self.count}.png", cv2.resize(mask.reshape(shape), dsize=(960, 540),
                                               interpolation=cv2.INTER_LINEAR).astype(np.uint8))

    def get_proposal_cls_probs(self, proposal_cls_probs, left_hand = None):
        if left_hand is None:
            proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
                                             num_classes=self.num_classes - 2 + 1).permute(0, 3, 1, 2)  # b, c, h, w
            proposal_cls_probs = proposal_cls_probs[:, :, :, :].mul(proposal_cls_one_hot)
        elif left_hand == True:
            proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
                                             num_classes=2).permute(0, 3, 1, 2)  # b, c, h, w
            proposal_cls_probs = proposal_cls_probs[:, :, :, :].mul(proposal_cls_one_hot)
        elif left_hand == False:
            proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
                                             num_classes=2).permute(0, 3, 1, 2)  # b, c, h, w
            proposal_cls_probs = proposal_cls_probs[:, :, :, :].mul(proposal_cls_one_hot)
        proposal_local_maximum_map = self.seek_local_maximum(proposal_cls_probs)  # b, c, h, w
        proposal_cls_probs = proposal_cls_probs + proposal_local_maximum_map
        return proposal_cls_probs
    def forward(self, x, pos_embeddings, targets=None):

        # proposal class logits
        # proposal_cls_logits = self.conv_proposal_cls_logits(x)  # b, c, h, w
          # b, c, h, w
        # proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
        #                                  num_classes=self.num_classes + 1).permute(0, 3, 1, 2)  # b, c, h, w
        # proposal_cls_probs = proposal_cls_probs.mul(proposal_cls_one_hot)
        # proposal_local_maximum_map = self.seek_local_maximum(proposal_cls_probs)  # b, c, h, w
        # proposal_cls_probs = proposal_cls_probs + proposal_local_maximum_map  # b, c, h, w

        proposal_left_cls_logits = self.conv_proposal_left_cls_logits(x)
        proposal_right_cls_logits = self.conv_proposal_right_cls_logits(x)
        proposal_obj_cls_logits = self.conv_proposal_obj_cls_logits(x)
        proposal_left_cls_probs = proposal_left_cls_logits.softmax(dim=1)
        proposal_right_cls_probs = proposal_right_cls_logits.softmax(dim=1)
        proposal_obj_cls_probs = proposal_obj_cls_logits.softmax(dim=1)
        proposal_left_cls_probs = self.get_proposal_cls_probs(proposal_left_cls_probs, left_hand=True)
        proposal_right_cls_probs = self.get_proposal_cls_probs(proposal_right_cls_probs, left_hand=False)
        proposal_obj_cls_probs = self.get_proposal_cls_probs(proposal_obj_cls_probs, left_hand=None)
        # contact map
        # contact_map, contact_map_queries, contact_map_pos_embeddings = self.contactmap(x)  # b, c, h, w

        # top-k indices
        topk_left_hand_indices = \
        torch.topk(proposal_left_cls_probs[:, :-1, :, :][:, None, :, :].flatten(2).max(1)[0],
                   self.topk // 2 - 10,
                   # 20,
                   dim=1)[1]  # b, q
        topk_right_hand_indices = \
        torch.topk(proposal_right_cls_probs[:, :-1, :, :][:, None, :, :].flatten(2).max(1)[0],
                   self.topk // 2 - 10,
                   # 20,
                   dim=1)[1]  # b, q
        topk_obj_indices = torch.topk(proposal_obj_cls_probs[:, :-1, :, :].flatten(2).max(1)[0],
                                      self.topk // 5, 
                                      dim=1)[
            1]  # b, q
        topk_indices = torch.concat((topk_left_hand_indices, topk_right_hand_indices, topk_obj_indices), dim=1)
        # topk_indices = torch.topk(proposal_cls_probs[:, :-1, :, :].flatten(2).max(1)[0], self.topk, dim=1)[1]  # b, q

        # self.save_mask(topk_left_hand_indices, proposal_obj_cls_probs.shape[-2:], type = 'left')
        # self.save_mask(topk_right_hand_indices, proposal_obj_cls_probs.shape[-2:], type = 'right')
        # self.save_mask(topk_obj_indices, proposal_obj_cls_probs.shape[-2:], type = 'obj')
        # self.save_mask(topk_indices, proposal_obj_cls_probs.shape[-2:], type='all')
        topk_indices = topk_indices.unsqueeze(1)  # b, 1, q

        # topk queries
        topk_proposals = torch.gather(x.flatten(2), dim=2, index=topk_indices.repeat(1, x.shape[1], 1))  # b, c, q
        pos_embeddings = pos_embeddings.repeat(x.shape[0], 1, 1, 1).flatten(2)
        topk_pos_embeddings = torch.gather(
            pos_embeddings, dim=2, index=topk_indices.repeat(1, pos_embeddings.shape[1], 1)
        )  # b, c, q

        # contact map
        contact_map = self.contactmap_estimation(x).view(-1, 2, 778, 1)
        contact_map_features = self.contactmap_conv(contact_map)
        topk_proposals += contact_map_features

        if self.training:
            locations = self.compute_coordinates(x).repeat(x.shape[0], 1, 1, 1)
            topk_locations = torch.gather(
                locations.flatten(2), dim=2, index=topk_indices.repeat(1, locations.shape[1], 1)
            )
            topk_locations = topk_locations.transpose(-1, -2)  # b, q, 2
        else:
            topk_locations = None

        # contactmap = torch.stack([t["contactmap"] for t in targets]).cuda()

        # top_k_values, top_k_indices = torch.topk(contactmap, self.contact_map_features // 2, dim=2)
        # top_k_values = top_k_values.view(contactmap.shape[0], -1)
        # top_k_indices = top_k_indices.view(contactmap.shape[0], self.contact_map_features)
        # top_k_values = inverse_sigmoid(top_k_values)
        # positional_encoding = torch.sin(top_k_indices.float()) + torch.cos(top_k_indices.float())
        #
        # # contactmap_ = contactmap.view(contactmap.shape[0], -1)
        # #
        # # contactmap_ = self.contactmap_features(contactmap_)
        # contact_queries = top_k_values.unsqueeze(2).repeat(1, 1, self.num_contact_queries).to(x.device)
        # positional_embeddings = positional_encoding.unsqueeze(2).repeat(1, 1, self.num_contact_queries).to(x.device)
        # Obtain positional embeddings for the topk embeddings
        # positional_indices = torch.arange(self.num_contact_queries).unsqueeze(0).repeat(contact_queries.size(0), 1).to(x.device)
        # positional_embeddings = self.positional_embedding(positional_indices)

        # positional_embeddings = None # positional_encoding(contactmap_.unsqueeze(2), 256).repeat(1, 1, self.num_contact_queries).to(x.device)

        proposal_cls_logits = [proposal_left_cls_logits, proposal_right_cls_logits, proposal_obj_cls_logits]
        return topk_proposals, topk_pos_embeddings, topk_locations, proposal_cls_logits, contact_map #, contact_queries, positional_embeddings


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ContactMapConvolution(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, num_queries, num_features
    ):
        super(ContactMapConvolution, self).__init__()

        self.num_queries = num_queries

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)

        # Linear layer to map the convolutional output to the desired dimensions
        self.linear = nn.Linear(out_channels, num_queries * num_features)

    def forward(self, x):
        # Assuming x is of shape [batch, 2, 778, 1]
        batch_size, _, num_vertices, _ = x.size()

        # Reshape to [batch, channels, sequence_length]
        x = x.view(batch_size, -1, num_vertices)

        # Apply 1D convolution
        x = self.conv1d(x)

        # Global average pooling over the sequence dimension
        x = torch.mean(x, dim=2)

        # Apply linear layer to get the final output
        output = self.linear(x)

        # Reshape the output to [batch, num_features, num_queries]
        output = output.view(batch_size, -1, self.num_queries)

        return output