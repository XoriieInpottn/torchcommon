#!/usr/bin/env python3

"""
@author: xi
@since: 2021-08-03
"""

import torch
from torch import nn
from torch.nn import functional as F


class DotProductAttention(nn.Module):

    def __init__(self, key_size: int, query_size: int, attention_size=None, norm=True):
        super(DotProductAttention, self).__init__()
        self._key_size = key_size
        self._query_size = query_size
        self._norm = norm

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor = None,
                return_value=True,
                return_score=False):
        # key: (n, s, k)
        # value: (n, s, v)
        if value is None:
            value = key

        assert len(key.shape) == 3 and key.shape[2] == self._key_size
        assert len(value.shape) == 3
        rank_query = len(query.shape)
        assert (rank_query == 2 and query.shape[1] == self._query_size) or \
               (rank_query == 3 and query.shape[2] == self._query_size)

        output_value = None
        output_score = None

        if rank_query == 2:
            # query is a single vector
            # query: (n, q)
            if self._norm:
                query = F.normalize(query, 2, 1)
                key = F.normalize(key, 2, 2)
            score = torch.einsum('nk,nsk->ns', query, key)
            score = F.softmax(score * 64, 1)  # (n, s)
            if return_score:
                output_score = score
            if return_value:
                output_value = score.unsqueeze(2) * value  # (n, s, v)
                output_value = output_value.sum(1)  # (n, v)
        elif rank_query == 3:
            # query is also a sequence
            # query: (n, l, q)
            if self._norm:
                query = F.normalize(query, 2, 2)
                key = F.normalize(key, 2, 2)
            score = torch.einsum('nlk,nsk->nls', query, key)
            score = F.softmax(score * 64, 2)  # (n, l, s)
            if return_score:
                output_score = score
            if return_value:
                output_value = (score.unsqueeze(3) * value.unsqueeze(1))  # (n, l, s, 1) * (n, 1, s, v) -> (n, l, s, v)
                output_value = output_value.sum(2)  # (n, l, v)
        else:
            raise RuntimeError('Invalid query shape.')

        return output_value, output_score


class BiLinearAttention(nn.Module):

    def __init__(self, key_size: int, query_size: int, bias=True):
        super(BiLinearAttention, self).__init__()
        self._key_size = key_size
        self._query_size = query_size
        self.layer = nn.Linear(
            in_features=query_size,
            out_features=key_size,
            bias=bias
        )

    def forward(self,
                key: torch.Tensor,
                query: torch.Tensor,
                value: torch.Tensor = None,
                return_value=True,
                return_score=False):
        # key: (n, s, k)
        # value: (n, s, v)
        # query: (n, q)
        if value is None:
            value = key

        assert len(key.shape) == 3 and key.shape[2] == self._key_size
        assert len(value.shape) == 3
        rank_query = len(query.shape)
        assert (rank_query == 2 and query.shape[1] == self._query_size) or \
               (rank_query == 3 and query.shape[2] == self._query_size)

        output_value = None
        output_score = None

        if rank_query == 2:
            # query is a single vector
            score = torch.einsum('nk,nsk->ns', self.layer(query), key)
            score = F.softmax(score, 1)  # (n, s)
            if return_score:
                output_score = score
            if return_value:
                output_value = score.unsqueeze(2) * value  # (n, s, v)
                output_value = output_value.sum(1)  # (n, v)
        elif rank_query == 3:
            # query is also a sequence
            # query: (n, l, q)
            score = torch.einsum('nlk,nsk->nls', self.layer(query), key)
            score = F.softmax(score, 2)  # (n, l, s)
            if return_score:
                output_score = score
            if return_value:
                output_value = (score.unsqueeze(3) * value.unsqueeze(1))  # (n, l, s, 1) * (n, 1, s, v) -> (n, l, s, v)
                output_value = output_value.sum(2)  # (n, l, v)
        else:
            raise RuntimeError('Invalid query shape.')

        return output_value, output_score


class MLPAttention(nn.Module):

    def __init__(self,
                 key_size: int,
                 query_size: int,
                 attention_size: int,
                 bias=True):
        super(MLPAttention, self).__init__()
        self._key_size = key_size
        self._query_size = query_size
        self._attention_size = attention_size
        self.key_layer = nn.Linear(
            in_features=key_size,
            out_features=attention_size,
            bias=bias
        )
        self.query_layer = nn.Linear(
            in_features=query_size,
            out_features=attention_size,
            bias=bias
        )
        self.relu = nn.ReLU(inplace=True)
        self.attention_layer = nn.Linear(
            in_features=attention_size,
            out_features=1,
            bias=bias
        )

    def forward(self,
                key: torch.Tensor,
                query: torch.Tensor,
                value: torch.Tensor = None,
                return_value=True,
                return_score=False):
        # key: (n, s, k)
        # value: (n, s, v)
        # query: (n, q)
        if value is None:
            value = key

        assert len(key.shape) == 3 and key.shape[2] == self._key_size
        assert len(value.shape) == 3
        rank_query = len(query.shape)
        assert (rank_query == 2 and query.shape[1] == self._query_size) or \
               (rank_query == 3 and query.shape[2] == self._query_size)

        output_value = None
        output_score = None

        if rank_query == 2:
            # query is a single vector
            # query: (n, q)
            h1 = self.query_layer(query)  # (n, a)
            h2 = self.key_layer(key)  # (n, s, a)
            h = h1.unsqueeze(1) + h2  # (n, s, a)
            h = self.relu(h)
            h = self.attention_layer(h)  # (n, s, 1)
            score = F.softmax(h, 1)
            if return_score:
                output_score = score.squeeze(2)
            if return_value:
                output_value = score * value  # (n, s, v)
                output_value = output_value.sum(1)  # (n, v)
        elif rank_query == 3:
            # query is also a sequence
            # query: (n, l, q)
            h1 = self.query_layer(query)  # (n, l, a)
            h2 = self.key_layer(key)  # (n, s, a)
            h = h1.unsqueeze(2) + h2.unsqueeze(1)  # (n, l, 1, a) + (n, 1, s, a) -> (n, l, s, a)
            h = self.relu(h)
            h = self.attention_layer(h)  # (n, l, s, 1)
            score = F.softmax(h, 2)
            if return_score:
                output_score = score.squeeze(2)
            if return_value:
                output_value = (score * value.unsqueeze(1))  # (n, l, s, 1) * (n, 1, s, v) -> (n, l, s, v)
                output_value = output_value.sum(2)  # (n, l, v)
        else:
            raise RuntimeError('Invalid query shape.')

        return output_value, output_score


def test():
    import numpy as np
    key = torch.tensor(np.random.uniform(-1, 1, (16, 10, 256)).astype(np.float32))
    query = torch.tensor(np.random.uniform(-1, 1, (16, 64)).astype(np.float32))
    bi_att = MLPAttention(256, 64, 100)
    output, score = bi_att(key, query, return_score=True)
    print(output.shape, score.shape)
    return 0


if __name__ == '__main__':
    raise SystemExit(test())
