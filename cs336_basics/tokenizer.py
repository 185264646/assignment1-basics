#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Yang Xiwen

import pickle
import regex
from typing import Iterable, Iterator, Self, cast
import itertools

def split_at_special_tokens(_input: str | int, special_tokens: list[str], reverse_vocab: dict[bytes, int]) -> list[str | int]:
    cur = [ _input ]
    for token in sorted(special_tokens, key = lambda b: len(b), reverse=True):
        new_cur: list[str | int] = []
        for i in cur:
            if isinstance(i, str):
                new_cur.extend(i.split(token))
            else:
                new_cur.append(i)
        cur = new_cur

        code = reverse_vocab[token.encode()]

        new_cur = []
        for i, j in itertools.pairwise(itertools.chain(cur, [0])):
            if isinstance(i, str) and isinstance(j, str):
                new_cur.extend((i, code))
            else:
                new_cur.append(i)
        cur = new_cur

    return cur

def pre_tokenization(_input: list[bytes]) -> Iterator[bytes]:
    GPT_PATTERN = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for i in _input:
        yield from regex.findall(GPT_PATTERN, i)

def iter_bytes(seq: bytes):
    for i in range(len(seq)):
        yield seq[i:i+1]

class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

    reverse_vocab: dict[bytes, int]

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [] if special_tokens is None else special_tokens

        self.reverse_vocab = { v: k for k, v in self.vocab.items() }


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath = str, special_tokens: list[str] | None = None) -> Self:
        with open(vocab_filepath, "rb") as f:
            buf = f.read()
            vocab = pickle.loads(buf)

        with open(merges_filepath, "rb") as f:
            buf = f.read()
            merges = pickle.loads(buf)

        return cls(vocab, merges, special_tokens)

    def _merge(self, token_list):
        bak = token_list
        for merge in self.merges:
            new_token_lst = []
            skip = False
            for i, j in itertools.pairwise(itertools.chain(token_list, [b''])):
                if skip:
                    skip = False
                    continue
                if (i, j) == merge:
                    new_token_lst.append(i + j)
                    skip = True
                else:
                    new_token_lst.append(i)
            token_list = new_token_lst

        bak[:] = token_list

    def encode(self, text: str) -> list[int]:
        rs = []
        split = split_at_special_tokens(text, self.special_tokens, self.reverse_vocab)
        for seg in split:
            if isinstance(seg, int):
                rs.append(seg)
            else:
                text = seg
                for pretoken in pre_tokenization([text.encode()]):
                    token_list = list(iter_bytes(pretoken))
                    self._merge(token_list)

                    rs.extend((self.reverse_vocab[i] for i in token_list))

        return rs

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for tmp in iterable:
            yield from self.encode(tmp)

    def decode(self, ids: list[int]) -> str:
        return b"".join((self.vocab[_id] for _id in ids)).decode(errors='replace')
