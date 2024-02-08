# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/transformer.py
import copy
import math
import numbers
import random
from functools import partial
from typing import List, Optional, Tuple, Union, Callable, Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import Linear, Module
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
from parrots.patched_mha_with_cache import multi_head_attention_forward_patched

_shape_t = Union[int, List[int], torch.Size]


def BalancedDoubleSwish(
        d_model, channel_dim=-1, max_abs=10.0, min_prob=0.25
) -> nn.Sequential:
    """
    ActivationBalancer -> DoubleSwish
    """
    balancer = ActivationBalancer(
        d_model, channel_dim=channel_dim, max_abs=max_abs, min_prob=min_prob
    )
    return nn.Sequential(
        balancer,
        DoubleSwish(),
    )

F.multi_head_attention_forward = multi_head_attention_forward_patched


# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/activation.py
class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use a special optimized implementation if all of the following
    conditions are met:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - dropout is 0
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - at most one of ``key_padding_mask`` or ``attn_mask`` is passed
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed

    If the optimized implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            linear1_cls=Linear,
            linear2_cls=Linear,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        if linear1_cls == Linear:
            if not self._qkv_same_embed_dim:
                self.q_proj_weight = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs)
                )
                self.k_proj_weight = Parameter(
                    torch.empty((embed_dim, self.kdim), **factory_kwargs)
                )
                self.v_proj_weight = Parameter(
                    torch.empty((embed_dim, self.vdim), **factory_kwargs)
                )
                self.register_parameter("in_proj_weight", None)
            else:
                self.in_proj_weight = Parameter(
                    torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
                )
                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

            if bias:
                self.in_proj_bias = Parameter(
                    torch.empty(3 * embed_dim, **factory_kwargs)
                )
            else:
                self.register_parameter("in_proj_bias", None)
            self.out_proj = NonDynamicallyQuantizableLinear(
                embed_dim, embed_dim, bias=bias, **factory_kwargs
            )

            self._reset_parameters()
        else:
            if not self._qkv_same_embed_dim:
                raise NotImplementedError
            else:
                self.in_proj_linear = linear1_cls(
                    embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs
                )
                self.in_proj_weight = self.in_proj_linear.weight

                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

                if bias:
                    self.in_proj_bias = self.in_proj_linear.bias
                else:
                    self.register_parameter("in_proj_bias", None)

            self.out_proj = linear2_cls(
                embed_dim, embed_dim, bias=bias, **factory_kwargs
            )

            if self.bias_k is not None:
                xavier_normal_(self.bias_k)
            if self.bias_v is not None:
                xavier_normal_(self.bias_v)

        self.add_zero_attn = add_zero_attn

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            cache=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(
                    key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        why_not_fast_path = ""
        if not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif (
                self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype
        ):
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = (
                "key_padding_mask is not supported with NestedTensor input"
            )
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                    [
                        (x is None or x.is_cuda or "cpu" in str(x.device))
                        for x in tensor_args
                    ]
            ):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(
                    [x is not None and x.requires_grad for x in tensor_args]
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1
                    if key_padding_mask is not None
                    else 0
                    if attn_mask is not None
                    else None,
                )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
                "MultiheadAttention does not support NestedTensor outside of its fast path. "
                + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                cache=cache,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                cache=cache,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
            self,
            normalized_shape: _shape_t,
            eps: float = 1e-5,
            elementwise_affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            self.bias = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                F.layer_norm(
                    input,
                    self.normalized_shape,
                    self.weight,
                    self.bias,
                    self.eps,
                ),
                embedding,
            )

        assert embedding is None
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class IdentityNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None,
    ) -> None:
        super(IdentityNorm, self).__init__()

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            return input

        assert embedding is None
        return input


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            return_layer_states: bool = False,
            cache=None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """
        if return_layer_states:
            layer_states = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    cache=cache,
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output

        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                cache=cache,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            batch_first: bool = False,
            norm_first: bool = False,
            device=None,
            dtype=None,
            linear1_self_attention_cls: nn.Module = nn.Linear,
            linear2_self_attention_cls: nn.Module = nn.Linear,
            linear1_feedforward_cls: nn.Module = nn.Linear,
            linear2_feedforward_cls: nn.Module = nn.Linear,
            layer_norm_cls: nn.Module = LayerNorm,
            layer_norm_eps: float = 1e-5,
            adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,  # 512 16
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )

        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model, **factory_kwargs
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)

        # # We can't test self.activation in forward() in TorchScript,
        # # so stash some information about it instead.
        # if activation is F.relu or isinstance(activation, torch.nn.ReLU):
        #     self.activation_relu_or_gelu = 1
        # elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
        #     self.activation_relu_or_gelu = 2
        # else:
        #     self.activation_relu_or_gelu = 0
        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            cache=None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                    src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding),
                src_mask,
                src_key_padding_mask,
                cache=cache,
            )
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, cache=cache),
                stage_embedding,
            )
            x = self.norm2(x + self._ff_block(x), stage_embedding)

        if is_src_tuple:
            return (x, stage_embedding)
        return x

    # self-attention block
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            cache=None,
    ) -> Tensor:
        # print(x.shape,attn_mask.shape,key_padding_mask)
        # torch.Size([1, 188, 512]) torch.Size([188, 188]) None
        # import os
        # os._exit(23333)
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            cache=cache,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = torch.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                dim=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)
    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        x_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        s = torch.sigmoid(x - 1.0)
        y = x * s

        if requires_grad:
            deriv = y * (1 - s) + s
            # notes on derivative of x * sigmoid(x - 1):
            # https://www.wolframalpha.com/input?i=d%2Fdx+%28x+*+sigmoid%28x-1%29%29
            # min \simeq -0.043638.  Take floor as -0.043637 so it's a lower bund
            # max \simeq 1.1990.   Take ceil to be 1.2 so it's an upper bound.
            # the combination of "+ torch.rand_like(deriv)" and casting to torch.uint8 (which
            # floors), should be expectation-preserving.
            floor = -0.043637
            ceil = 1.2
            d_scaled = (deriv - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                deriv
            )
            if __name__ == "__main__":
                # for self-testing only.
                assert d_scaled.min() >= 0.0
                assert d_scaled.max() < 256.0
            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
        if x.dtype == torch.float16 or torch.is_autocast_enabled():
            y = y.to(torch.float16)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.043637
        ceil = 1.2
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


class ActivationBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            x: Tensor,
            scale_factor: Tensor,
            sign_factor: Optional[Tensor],
            channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        xgt0 = x > 0
        if sign_factor is None:
            ctx.save_for_backward(xgt0, scale_factor)
        else:
            ctx.save_for_backward(xgt0, scale_factor, sign_factor)
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None]:
        if len(ctx.saved_tensors) == 3:
            xgt0, scale_factor, sign_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
                sign_factor = sign_factor.unsqueeze(-1)
            factor = sign_factor + scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        else:
            xgt0, scale_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
            factor = scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        neg_delta_grad = x_grad.abs() * factor
        return (
            x_grad - neg_delta_grad,
            None,
            None,
            None,
        )


def _compute_scale_factor(
        x: Tensor,
        channel_dim: int,
        min_abs: float,
        max_abs: float,
        gain_factor: float,
        max_factor: float,
) -> Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    x_abs_mean = torch.mean(x.abs(), dim=sum_dims).to(torch.float32)

    if min_abs == 0.0:
        below_threshold = 0.0
    else:
        # below_threshold is 0 if x_abs_mean > min_abs, can be at most max_factor if
        # x_abs)_mean , min_abs.
        below_threshold = ((min_abs - x_abs_mean) * (gain_factor / min_abs)).clamp(
            min=0, max=max_factor
        )

    above_threshold = ((x_abs_mean - max_abs) * (gain_factor / max_abs)).clamp(
        min=0, max=max_factor
    )

    return below_threshold - above_threshold


def _compute_sign_factor(
        x: Tensor,
        channel_dim: int,
        min_positive: float,
        max_positive: float,
        gain_factor: float,
        max_factor: float,
) -> Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    proportion_positive = torch.mean((x > 0).to(torch.float32), dim=sum_dims)
    if min_positive == 0.0:
        factor1 = 0.0
    else:
        # 0 if proportion_positive >= min_positive, else can be
        # as large as max_factor.
        factor1 = (
                (min_positive - proportion_positive) * (gain_factor / min_positive)
        ).clamp_(min=0, max=max_factor)

    if max_positive == 1.0:
        factor2 = 0.0
    else:
        # 0 if self.proportion_positive <= max_positive, else can be
        # as large as -max_factor.
        factor2 = (
                (proportion_positive - max_positive) * (gain_factor / (1.0 - max_positive))
        ).clamp_(min=0, max=max_factor)
    sign_factor = factor1 - factor2
    # require min_positive != 0 or max_positive != 1:
    assert not isinstance(sign_factor, float)
    return sign_factor


class ActivationBalancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.

    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           max_factor: the maximum factor by which we modify the derivatives for
              either the sign constraint or the magnitude constraint;
              e.g. with max_factor=0.02, the the derivatives would be multiplied by
              values in the range [0.98..1.02].
           sign_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_positive and max_positive
              are violated.
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
          min_prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.  Early in training we may use
             higher probabilities than this; it will decay to this value.
    """

    def __init__(
            self,
            num_channels: int,
            channel_dim: int,
            min_positive: float = 0.05,
            max_positive: float = 0.95,
            max_factor: float = 0.04,
            sign_gain_factor: float = 0.01,
            scale_gain_factor: float = 0.02,
            min_abs: float = 0.2,
            max_abs: float = 100.0,
            min_prob: float = 0.1,
    ):
        super(ActivationBalancer, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.min_prob = min_prob
        self.sign_gain_factor = sign_gain_factor
        self.scale_gain_factor = scale_gain_factor

        # count measures how many times the forward() function has been called.
        # We occasionally sync this to a tensor called `count`, that exists to
        # make sure it is synced to disk when we load and save the model.
        self.cpu_count = 0
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> Tensor:
        count = self.cpu_count
        self.cpu_count += 1

        if random.random() < 0.01:
            # Occasionally sync self.cpu_count with self.count.
            # count affects the decay of 'prob'.  don't do this on every iter,
            # because syncing with the GPU is slow.
            self.cpu_count = max(self.cpu_count, self.count.item())
            self.count.fill_(self.cpu_count)

        # the prob of doing some work exponentially decreases from 0.5 till it hits
        # a floor at min_prob (==0.1, by default)
        prob = max(self.min_prob, 0.5 ** (1 + (count / 4000.0)))

        if random.random() < prob:
            sign_gain_factor = 0.5
            if self.min_positive != 0.0 or self.max_positive != 1.0:
                sign_factor = _compute_sign_factor(
                    x,
                    self.channel_dim,
                    self.min_positive,
                    self.max_positive,
                    gain_factor=self.sign_gain_factor / prob,
                    max_factor=self.max_factor,
                )
            else:
                sign_factor = None

            scale_factor = _compute_scale_factor(
                x.detach(),
                self.channel_dim,
                min_abs=self.min_abs,
                max_abs=self.max_abs,
                gain_factor=self.scale_gain_factor / prob,
                max_factor=self.max_factor,
            )
            return ActivationBalancerFunction.apply(
                x,
                scale_factor,
                sign_factor,
                self.channel_dim,
            )
        else:
            return x


# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/embedding.py
class TokenEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            vocab_size: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index: index + 1]

    def forward(self, x: torch.Tensor):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        return x


class SinePositionalEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            dropout: float = 0.0,
            scale: bool = False,
            alpha: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.embedding_dim)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(output)


# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/model/utils.py\
def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    #>>> lengths = torch.tensor([1, 3, 2, 5])
    #>>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
        logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


def multinomial_sample_one_no_sync(
        probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
        logits,
        previous_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        repetition_penalty: float = 1.0,
):
    previous_tokens = previous_tokens.squeeze()
    # print(logits.shape,previous_tokens.shape)
    # pdb.set_trace()
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
        logits,
        previous_tokens: Optional[torch.Tensor] = None,
        **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs
