from __future__ import annotations
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class RelativePositionBias2D(nn.Module):
    """
    2D relative position bias for attention on image patch grids.
    Supports self-attention and cross-attention between 2D token grids.
    """
    def __init__(self, num_heads: int, max_pos_size: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.max_pos_size = max_pos_size
        table_size = 2 * max_pos_size - 1

        self.rel_height = nn.Parameter(torch.zeros(table_size, num_heads))
        self.rel_width = nn.Parameter(torch.zeros(table_size, num_heads))

        nn.init.normal_(self.rel_height, std=0.02)
        nn.init.normal_(self.rel_width, std=0.02)

    def _relative_index(self, q_size: int, k_size: int, device: torch.device) -> Tensor:
        q_coords = torch.arange(q_size, device=device)[:, None]
        k_coords = torch.arange(k_size, device=device)[None, :]
        rel = q_coords - k_coords + self.max_pos_size - 1
        rel = rel.clamp(0, 2 * self.max_pos_size - 2)
        return rel.long()

    def forward(
        self,
        q_hw: tuple[int, int],
        k_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        qh, qw = q_hw
        kh, kw = k_hw

        max_need = max(qh, qw, kh, kw)
        if max_need > self.max_pos_size:
            raise ValueError(
                f"Current token grid {max_need} exceeds max_pos_size={self.max_pos_size}. "
                f"Increase --max_rel_pos."
            )

        idx_h = self._relative_index(qh, kh, device)  # [qh, kh]
        idx_w = self._relative_index(qw, kw, device)  # [qw, kw]

        # [qh, kh, heads]
        bias_h = self.rel_height[idx_h]
        # [qw, kw, heads]
        bias_w = self.rel_width[idx_w]

        # -> [qh, qw, kh, kw, heads]
        bias = bias_h[:, None, :, None, :] + bias_w[None, :, None, :, :]

        # -> [heads, qh*qw, kh*kw]
        bias = bias.permute(4, 0, 1, 2, 3).reshape(self.num_heads, qh * qw, kh * kw)
        return bias.to(dtype=dtype)


class MultiHeadAttentionRelPos(nn.Module):
    """
    Drop-in attention layer with optional 2D relative position bias.
    Input/Output format: [N, B, C]
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_rel_pos: bool = False,
        max_pos_size: int = 64,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.use_rel_pos = use_rel_pos
        self.rel_pos = RelativePositionBias2D(nhead, max_pos_size) if use_rel_pos else None

    def _merge_masks(
        self,
        base_bias: Optional[Tensor],
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        batch_size: int,
        num_queries: int,
        num_keys: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        bias = base_bias

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                float_mask = torch.zeros(
                    (1, 1, num_queries, num_keys), device=device, dtype=dtype
                )
                float_mask = float_mask.masked_fill(attn_mask[None, None, :, :], float("-inf"))
            else:
                float_mask = attn_mask.to(device=device, dtype=dtype)
                if float_mask.dim() == 2:
                    float_mask = float_mask[None, None, :, :]
                elif float_mask.dim() == 3:
                    # expected [B*H, Nq, Nk]
                    float_mask = float_mask.view(batch_size, self.nhead, num_queries, num_keys)
            bias = float_mask if bias is None else bias + float_mask

        if key_padding_mask is not None:
            pad = key_padding_mask.to(device=device).bool()[:, None, None, :]  # [B,1,1,Nk]
            if bias is None:
                bias = torch.zeros(
                    (batch_size, self.nhead, num_queries, num_keys),
                    device=device,
                    dtype=dtype,
                )
            elif bias.size(0) == 1 and batch_size > 1:
                bias = bias.expand(batch_size, -1, -1, -1).clone()

            bias = bias.masked_fill(pad, float("-inf"))

        return bias

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_hw: tuple[int, int],
        k_hw: tuple[int, int],
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # query/key/value: [N, B, C]
        nq, bsz, _ = query.shape
        nk, _, _ = key.shape

        q = query.permute(1, 0, 2)  # [B, Nq, C]
        k = key.permute(1, 0, 2)    # [B, Nk, C]
        v = value.permute(1, 0, 2)  # [B, Nk, C]

        q = self.q_proj(q).view(bsz, nq, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).view(bsz, nk, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).view(bsz, nk, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        rel_bias = None
        if self.use_rel_pos:
            rel_bias = self.rel_pos(
                q_hw=q_hw,
                k_hw=k_hw,
                device=query.device,
                dtype=q.dtype,
            )  # [H, Nq, Nk]
            rel_bias = rel_bias.unsqueeze(0)  # [1, H, Nq, Nk]

        full_mask = self._merge_masks(
            base_bias=rel_bias,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            batch_size=bsz,
            num_queries=nq,
            num_keys=nk,
            device=query.device,
            dtype=q.dtype,
        )

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=full_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, H, Nq, D]

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(bsz, nq, self.d_model)
        attn_out = self.out_proj(attn_out)

        return attn_out.permute(1, 0, 2)  # [Nq, B, C]


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        use_rel_pos=True,
        max_pos_size=64,
    ):
        super().__init__()

        # Style encoder: no relative positional bias (closer to original StyTr² spirit)
        style_encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
            use_rel_pos=False,
            max_pos_size=max_pos_size,
        )
        style_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_s = TransformerEncoder(style_encoder_layer, num_encoder_layers, style_encoder_norm)

        # Content encoder: relative position attention replaces CAPE
        content_encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
            use_rel_pos=use_rel_pos,
            max_pos_size=max_pos_size,
        )
        content_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_c = TransformerEncoder(content_encoder_layer, num_encoder_layers, content_encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
            use_rel_pos=use_rel_pos,
            max_pos_size=max_pos_size,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.use_rel_pos = use_rel_pos
        self.max_pos_size = max_pos_size

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style, mask, content, pos_embed_c, pos_embed_s):
        """
        style/content: [B, C, H, W] after PatchEmbed
        Returns: [B, C, H, W]
        """
        bsz, c, hs, ws = style.shape
        _, _, hc, wc = content.shape

        # Relative Position Attention replaces CAPE and absolute position injection.
        # If you want hybrid absolute+relative, remove these two lines.
        pos_embed_c = None
        pos_embed_s = None

        style = style.flatten(2).permute(2, 0, 1)    # [Ns, B, C]
        content = content.flatten(2).permute(2, 0, 1)  # [Nc, B, C]

        style = self.encoder_s(
            style,
            src_key_padding_mask=mask,
            pos=pos_embed_s,
            hw=(hs, ws),
        )
        content = self.encoder_c(
            content,
            src_key_padding_mask=mask,
            pos=pos_embed_c,
            hw=(hc, wc),
        )

        hs_out = self.decoder(
            content,
            style,
            memory_key_padding_mask=mask,
            pos=pos_embed_s,
            query_pos=pos_embed_c,
            tgt_hw=(hc, wc),
            memory_hw=(hs, ws),
        )[0]  # [Nc, B, C]

        hs_out = hs_out.permute(1, 2, 0).contiguous().view(bsz, c, hc, wc)
        return hs_out


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        hw: Optional[tuple[int, int]] = None,
    ):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
                hw=hw,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        tgt_hw: Optional[tuple[int, int]] = None,
        memory_hw: Optional[tuple[int, int]] = None,
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                tgt_hw=tgt_hw,
                memory_hw=memory_hw,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm is not None else output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        use_rel_pos=False,
        max_pos_size=64,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionRelPos(
            d_model, nhead, dropout=dropout,
            use_rel_pos=use_rel_pos,
            max_pos_size=max_pos_size,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        hw: Optional[tuple[int, int]] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, src,
            q_hw=hw, k_hw=hw,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        hw: Optional[tuple[int, int]] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, src2,
            q_hw=hw, k_hw=hw,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        hw: Optional[tuple[int, int]] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, hw)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, hw)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        use_rel_pos=False,
        max_pos_size=64,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionRelPos(
            d_model, nhead, dropout=dropout,
            use_rel_pos=use_rel_pos,
            max_pos_size=max_pos_size,
        )
        self.multihead_attn = MultiHeadAttentionRelPos(
            d_model, nhead, dropout=dropout,
            use_rel_pos=use_rel_pos,
            max_pos_size=max_pos_size,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        tgt_hw: Optional[tuple[int, int]] = None,
        memory_hw: Optional[tuple[int, int]] = None,
    ):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        tgt2 = self.self_attn(
            q, k, v,
            q_hw=tgt_hw, k_hw=tgt_hw,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            q_hw=tgt_hw,
            k_hw=memory_hw,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        tgt_hw: Optional[tuple[int, int]] = None,
        memory_hw: Optional[tuple[int, int]] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, tgt2,
            q_hw=tgt_hw, k_hw=tgt_hw,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            q_hw=tgt_hw,
            k_hw=memory_hw,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        tgt_hw: Optional[tuple[int, int]] = None,
        memory_hw: Optional[tuple[int, int]] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                memory_key_padding_mask, pos, query_pos, tgt_hw, memory_hw
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
            memory_key_padding_mask, pos, query_pos, tgt_hw, memory_hw
        )


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def build_transformer(args):
    return Transformer(
        d_model=getattr(args, "hidden_dim", 512),
        dropout=getattr(args, "dropout", 0.1),
        nhead=getattr(args, "nheads", 8),
        dim_feedforward=getattr(args, "dim_feedforward", 2048),
        num_encoder_layers=getattr(args, "enc_layers", 3),
        num_decoder_layers=getattr(args, "dec_layers", 3),
        normalize_before=getattr(args, "pre_norm", False),
        return_intermediate_dec=True,
        use_rel_pos=getattr(args, "use_rel_pos_attention", True),
        max_pos_size=getattr(args, "max_rel_pos", 64),
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")