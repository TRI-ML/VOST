import torch.nn.functional as F
from torch import nn
import torch

from networks.layers.basic import DropPath, GroupNorm1D, GNActDWConv2d, seq_to_2d
from networks.layers.attention import MultiheadAttention, MultiheadLocalAttentionV2, MultiheadLocalAttentionV3


def _get_norm(indim, type='ln', groups=8):
    if type == 'gn' and groups != 1:
        return GroupNorm1D(indim, groups)
    elif type == 'gn' and groups == 1:
        return nn.GroupNorm(groups, indim)
    else:
        return nn.LayerNorm(indim)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        F"activation should be relu/gele/glu, not {activation}.")


class LongShortTermTransformer(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 self_nhead=8,
                 att_nhead=8,
                 dim_feedforward=1024,
                 emb_dropout=0.,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 droppath_scaling=False,
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True,
                 simplified=False,
                 stopgrad=False,
                 joint_longatt=False,
                 linear_q=False,
                 recurrent_stm=False,
                 norm_inp=False,
                 recurrent_ltm=False):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)

        layers = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            if simplified:
                layers.append(
                    SimplifiedTransformerBlock(d_model, self_nhead, att_nhead,
                                                dim_feedforward, droppath_rate,
                                                activation, stopgrad=stopgrad, joint_longatt=joint_longatt, linear_q=linear_q, recurrent_stm=recurrent_stm))
            elif recurrent_ltm:
                layers.append(RecurrentLongNormalShortBlock(d_model, self_nhead, att_nhead,
                                                dim_feedforward, droppath_rate,
                                                activation))
            else:
                layers.append(
                    LongShortTermTransformerBlock(d_model, self_nhead, att_nhead,
                                                  dim_feedforward, droppath_rate,
                                                  lt_dropout, st_dropout,
                                                  droppath_lst, activation))
        self.layers = nn.ModuleList(layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model, type='ln') for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)

    def forward(self,
                tgt,
                long_term_memories,
                short_term_memories,
                curr_id_emb=None,
                self_pos=None,
                size_2d=None):

        output = self.emb_dropout(tgt)

        intermediate = []
        intermediate_memories = []

        for idx, layer in enumerate(self.layers):
            output, memories = layer(output,
                                     long_term_memories[idx] if
                                     long_term_memories is not None else None,
                                     short_term_memories[idx] if
                                     short_term_memories is not None else None,
                                     curr_id_emb=curr_id_emb,
                                     self_pos=self_pos,
                                     size_2d=size_2d)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)

        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](
                            intermediate[idx])

        if self.return_intermediate:
            return intermediate, intermediate_memories

        return output, memories


class LongShortTermTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True):
        super().__init__()

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.norm2 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False,
                                                 dropout=lt_dropout)
        if enable_corr:
            try:
                import spatial_correlation_sampler
                MultiheadLocalAttention = MultiheadLocalAttentionV2
            except Exception as inst:
                print(inst)
                print(
                    "Failed to import PyTorch Correlation. For better efficiency, please install it."
                )
                MultiheadLocalAttention = MultiheadLocalAttentionV3
        else:
            MultiheadLocalAttention = MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model,
                                                       att_nhead,
                                                       dilation=local_dilation,
                                                       use_linear=False,
                                                       dropout=st_dropout)

        self.droppath_lst = droppath_lst

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K = curr_K
            global_V = self.linear_V(curr_V + curr_id_emb)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + tgt2 + tgt3

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class RecurrentLongNormalShortBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True):
        super().__init__()

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.norm2 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_QMem = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False)
        if enable_corr:
            try:
                import spatial_correlation_sampler
                MultiheadLocalAttention = MultiheadLocalAttentionV2
            except Exception as inst:
                print(inst)
                print(
                    "Failed to import PyTorch Correlation. For better efficiency, please install it."
                )
                MultiheadLocalAttention = MultiheadLocalAttentionV3
        else:
            MultiheadLocalAttention = MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model,
                                                       att_nhead,
                                                       dilation=local_dilation,
                                                       use_linear=False)

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K = curr_K
            global_V = self.linear_V(curr_V + curr_id_emb)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        tgt2 = self.long_term_attn(curr_Q, torch.cat((global_K, curr_K), 0), torch.cat((global_V, curr_V), 0))[0]
        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        global_K = self.linear_QMem(tgt2)
        global_V = tgt2
        if curr_id_emb is not None:
            global_V = self.linear_V(global_V + curr_id_emb)

        tgt = tgt + tgt2 + tgt3

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class SimplifiedTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 activation="gelu",
                 stopgrad=False,
                 joint_longatt=False,
                 linear_q=False,
                 recurrent_stm=False):
        super().__init__()

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.norm2 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_QMem = nn.Linear(d_model, d_model)
        self.linear_VMem = nn.Linear(d_model, d_model)
        if not linear_q:
            self.norm4 = _get_norm(d_model)

        self.linear_KMem = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False)
        
        if not stopgrad:
            self.short_term_attn = MultiheadAttention(d_model,
                                                        att_nhead,
                                                        use_linear=False)
        else:
            try:
                import spatial_correlation_sampler
                MultiheadLocalAttention = MultiheadLocalAttentionV2
            except Exception as inst:
                print(inst)
                print(
                    "Failed to import PyTorch Correlation. For better efficiency, please install it."
                )
                MultiheadLocalAttention = MultiheadLocalAttentionV3
            self.short_term_attn = MultiheadLocalAttention(d_model,
                                                        att_nhead,
                                                        dilation=1,
                                                        use_linear=False)

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self.stopgrad = stopgrad
        self.joint_longatt = joint_longatt
        self.linear_q = linear_q
        self.recurrent_stm = recurrent_stm
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = curr_Q

        if curr_id_emb is not None:
            global_K = curr_K
            global_V = self.linear_V(curr_V + curr_id_emb)
            local_K = global_K
            local_V = global_V
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        if self.joint_longatt:
            tgt2 = self.long_term_attn(curr_Q, torch.cat((global_K, curr_K), 0), torch.cat((global_V, curr_V), 0))[0]
        else:
            tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        
        if self.linear_q:
            tgt3 = self.short_term_attn(local_Q, torch.cat((local_K, curr_K), 0), torch.cat((local_V, curr_V), 0))[0]
        else:
            if self.stopgrad:
                K = local_K + curr_K
                # K = self.norm4(K)
                V = local_V + curr_V
                # V = self.norm4(V)
                tgt3 = self.short_term_attn(seq_to_2d(local_Q, size_2d), seq_to_2d(K, size_2d), seq_to_2d(V, size_2d))[0]
            else:
                tgt3 = self.short_term_attn(local_Q, self.norm4(local_K + curr_K), self.norm4(local_V + curr_V))[0]

        if self.recurrent_stm:
            _tgt3 = tgt3
            
            local_K = self.linear_QMem(_tgt3)
            local_V = _tgt3
            if curr_id_emb is not None:
                local_V = self.linear_VMem(local_V + curr_id_emb)

        tgt = tgt + tgt2 + tgt3

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)