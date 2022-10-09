#!/usr/bin/env python3

"""
@author: xi
@since: 2021-08-03
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .functional import softmax_with_mask


class BiLinearAttention(nn.Module):

    def __init__(self,
                 query_size: int,
                 key_size: int):
        super(BiLinearAttention, self).__init__()
        self._query_size = query_size
        self._key_size = key_size

        self.weight = nn.Parameter(torch.Tensor(query_size, key_size))

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor = None,
                mask=None,
                *,
                score_only=False):
        # query: (n, q) or (n, l, q)
        # key: (n, s, k)
        # value: (n, s, v)
        if value is None:
            value = key

        assert len(key.shape) == 3 and key.shape[2] == self._key_size
        assert len(value.shape) == 3
        rank_query = len(query.shape)
        assert (rank_query == 2 and query.shape[1] == self._query_size) or \
               (rank_query == 3 and query.shape[2] == self._query_size)

        if rank_query == 2:
            # query is a single vector
            # query: (n, q)
            # key: (n, s, k)
            # value: (n, s, v)
            score = torch.einsum('nk,nsk->ns', query @ self.weight, key)
            score = softmax_with_mask(score, 1, mask=mask)  # (n, s)
            if score_only:
                return score
            else:
                score_ = score.unsqueeze(2)  # (n, s, 1)
                output_value = score_ * value  # (n, s, v)
                output_value = output_value.sum(1)  # (n, v)
                return output_value, score
        elif rank_query == 3:
            # query is also a sequence
            # query: (n, l, q)
            # key: (n, s, k)
            # value: (n, s, v)
            score = torch.einsum('nlk,nsk->nls', query @ self.weight, key)
            score = softmax_with_mask(score, 2, mask=mask)  # (n, l, s)
            if score_only:
                return score
            else:
                score_ = score.unsqueeze(3)  # (n, l, s, 1)
                value_ = value.unsqueeze(1)  # (n, 1, s, v)
                output_value = score_ * value_  # (n, l, s, 1) * (n, 1, s, v) -> (n, l, s, v)
                output_value = output_value.sum(2)  # (n, l, v)
                return output_value, score
        else:
            raise RuntimeError('Invalid query shape.')


class MLPAttention(nn.Module):

    def __init__(self,
                 query_size: int,
                 key_size: int,
                 attention_size: int,
                 bias=True,
                 norm=True):
        super(MLPAttention, self).__init__()
        self._query_size = query_size
        self._key_size = key_size
        self._attention_size = attention_size

        self.query_layer = nn.Linear(
            in_features=query_size,
            out_features=attention_size,
            bias=bias
        )
        self.key_layer = nn.Linear(
            in_features=key_size,
            out_features=attention_size,
            bias=bias
        )
        self.norm = nn.LayerNorm(attention_size) if norm else None
        self.non_lin = nn.ReLU(inplace=True)
        self.attention_layer = nn.Linear(
            in_features=attention_size,
            out_features=1,
            bias=bias
        )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor = None,
                mask=None,
                *,
                score_only=False):
        # query: (n, q) or (n, l, q)
        # key: (n, s, k)
        # value: (n, s, v)
        if value is None:
            value = key

        assert len(key.shape) == 3 and key.shape[2] == self._key_size
        assert len(value.shape) == 3
        rank_query = len(query.shape)
        assert (rank_query == 2 and query.shape[1] == self._query_size) or \
               (rank_query == 3 and query.shape[2] == self._query_size)

        if rank_query == 2:
            # query is a single vector
            # query: (n, q)
            # key: (n, s, k)
            # value: (n, s, v)
            h1 = self.query_layer(query)  # (n, a)
            h2 = self.key_layer(key)  # (n, s, a)
            h = h1.unsqueeze(1) + h2  # (n, s, a)
            if self.norm is not None:
                h = self.norm(h)
            h = self.non_lin(h)
            h = self.attention_layer(h)  # (n, s, 1)
            score = softmax_with_mask(h, 1, mask=mask)
            if score_only:
                return score.squeeze(2)
            else:
                output_value = score * value  # (n, s, v)
                output_value = output_value.sum(1)  # (n, v)
                return output_value, score.squeeze(2)
        elif rank_query == 3:
            # query is also a sequence
            # query: (n, l, q)
            # key: (n, s, k)
            # value: (n, s, v)
            h1 = self.query_layer(query)  # (n, l, a)
            h2 = self.key_layer(key)  # (n, s, a)
            h = h1.unsqueeze(2) + h2.unsqueeze(1)  # (n, l, 1, a) + (n, 1, s, a) -> (n, l, s, a)
            if self.norm is not None:
                h = self.norm(h)
            h = self.non_lin(h)
            h = self.attention_layer(h)  # (n, l, s, 1)
            score = softmax_with_mask(h, 2, mask=mask)
            if score_only:
                return score.squeeze(2)
            else:
                output_value = (score * value.unsqueeze(1))  # (n, l, s, 1) * (n, 1, s, v) -> (n, l, s, v)
                output_value = output_value.sum(2)  # (n, l, v)
                return output_value, score.squeeze(2)
        else:
            raise RuntimeError('Invalid query shape.')


class DotProductAttention(nn.Module):

    def __init__(self, size: int, scale=None, normalize=False):
        super(DotProductAttention, self).__init__()
        self._size = size
        self._scale = math.sqrt(size) if scale is None else scale
        self._normalize = normalize

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor = None,
                mask=None,
                *,
                score_only=False, ):
        # query: (n, q) or (n, l, q)
        # key: (n, s, k)
        # value: (n, s, v)
        if value is None:
            value = key

        assert len(key.shape) == 3 and key.shape[2] == self._size
        assert len(value.shape) == 3
        rank_query = len(query.shape)
        assert (rank_query == 2 and query.shape[1] == self._size) or \
               (rank_query == 3 and query.shape[2] == self._size)

        if rank_query == 2:
            # query is a single vector
            # query: (n, q)
            # key: (n, s, k)
            # value: (n, s, v)
            if self._normalize:
                query = F.normalize(query, 2, 1)
                key = F.normalize(key, 2, 2)
            score = torch.einsum('nk,nsk->ns', query, key)
            score = softmax_with_mask(score / self._scale, 1, mask=mask)  # (n, s)
            if score_only:
                return score
            else:
                output_value = score.unsqueeze(2) * value  # (n, s, v)
                output_value = output_value.sum(1)  # (n, v)
                return output_value, score
        elif rank_query == 3:
            # query is also a sequence
            # query: (n, l, q)
            # key: (n, s, k)
            # value: (n, s, v)
            if self._normalize:
                query = F.normalize(query, 2, 2)
                key = F.normalize(key, 2, 2)
            score = torch.einsum('nlk,nsk->nls', query, key)
            score = softmax_with_mask(score / self._scale, 2, mask=mask)  # (n, l, s)
            if score_only:
                return score
            else:
                output_value = (score.unsqueeze(3) * value.unsqueeze(1))  # (n, l, s, 1) * (n, 1, s, v) -> (n, l, s, v)
                output_value = output_value.sum(2)  # (n, l, v)
                return output_value, score
        else:
            raise RuntimeError('Invalid query shape.')
