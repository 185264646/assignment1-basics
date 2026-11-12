#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Yang Xiwen
import regex
import itertools
from collections import Counter
from typing import List, Tuple, Dict, Set

def read_to_buffer(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def split_at_special_tokens(_input: str, special_tokens: List[str]) -> List[str]:
    cur = [ _input ]
    for token in special_tokens:
        cur = [ i for i in cur for i in i.split(token) ]
    return cur

def pre_tokenization(_input: List[bytes]):
    GPT_PATTERN = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for i in _input:
        yield from regex.findall(GPT_PATTERN, i)

def build_initial_vocab(_input, special_tokens: List[str]):
    init = { idx: bytes([idx]) for idx in range(256) }
    init.update({ idx + 256: val.encode() for idx, val in enumerate(special_tokens) })
    return init

PairCount = Dict[Tuple[bytes, bytes], int]
PairToAffectedTokens = Dict[Tuple[bytes, bytes], Set[int]]

def build_pairs(token_list: List[List[bytes]], count_list: List[int]) -> Tuple[PairCount, PairToAffectedTokens]:
    pair_count: PairCount = {}
    pair_to_affected_tokens: PairToAffectedTokens = {}
    for idx, token in enumerate(token_list):
        addition = count_list[idx]
        for pair in itertools.pairwise(token):
            pair_count.setdefault(pair, 0)
            pair_to_affected_tokens.setdefault(pair, set())
            pair_count[pair] += addition
            pair_to_affected_tokens[pair].add(idx)

    return (pair_count, pair_to_affected_tokens)

def merge(lst: List[List[bytes]], pair: Tuple[bytes, bytes], update_list: Set[int]):
    # TODO: use itertools
    skip = False
    cnt = 0
    for token in (lst[i] for i in update_list):
        tmp = []
        for first, second in itertools.pairwise(itertools.chain(token, [b''])):
            if skip:
                cnt += 1
                skip = False
                continue
            if (first, second) == pair:
                tmp.append(pair[0] + pair[1])
                skip = True
            else:
                tmp.append(first)
        token[:] = tmp
    #print(f"merged {cnt} times")
    return lst

def iter_bytes(b: bytes):
    for i in range(len(b)):
        yield b[i:i+1]

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
    buffer = read_to_buffer(input_path)
    split = split_at_special_tokens(buffer, special_tokens)
    split_as_bytes = [ i.encode() for i in split ]
    pretoken_counter = Counter(pre_tokenization(split_as_bytes))
    #print(f"{len(pretoken_counter)=}")
    token_list = [ [ b for b in iter_bytes(token) ] for token in pretoken_counter.keys() ]
    count_list = [ pretoken_counter[b''.join(token)] for token in token_list ]
    vocab = build_initial_vocab(token_list, special_tokens)
    merges = []
    while len(vocab) < vocab_size:
        pair_count, pair_to_affected_tokens = build_pairs(token_list, count_list)
        pair = max(pair_count.items(), key = lambda p: (p[1], p[0]))
        merges.append((pair[0][0], pair[0][1]))
        token_list = merge(token_list, pair[0], pair_to_affected_tokens[pair[0]])
        vocab[len(vocab)] = b''.join(pair[0])
    return (vocab, merges)
