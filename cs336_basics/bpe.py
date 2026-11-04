#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Yang Xiwen
import regex
import itertools
from collections import Counter

def read_to_buffer(path):
    with open(path, "r") as f:
        return f.read()

def split_at_special_tokens(_input, special_tokens):
    cur = [ _input ]
    for token in special_tokens:
        cur = [ i for i in cur for i in i.split(token) ]
    return cur

def pre_tokenization(_input):
    GPT_PATTERN = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for i in _input:
        yield from regex.findall(GPT_PATTERN, i)

def build_initial_vocab(_input, special_tokens):
    init = { idx: bytes([idx]) for idx in range(256) }
    init.update({ idx + 256: val.encode() for idx, val in enumerate(special_tokens) })
    return init

def build_pairs(token_list, pretoken_counter):
    _dict = {}
    for token in token_list:
        key = b''.join(token)
        addition = pretoken_counter[key]
        #print(key)
        assert addition > 0
        for pair in itertools.pairwise(token):
            _dict.setdefault(pair, 0)
            _dict[pair] += addition

    return _dict

def merge(lst, pair):
    # TODO: use itertools
    rs = []
    skip = False
    cnt = 0
    for token in lst:
        tmp = []
        for first, second in itertools.pairwise(itertools.chain(token, [None])):
            if skip:
                cnt += 1
                skip = False
                continue
            if (first, second) == pair:
                tmp.append(pair[0] + pair[1])
                skip = True
            else:
                tmp.append(first)
        rs.append(tmp)
    #print(f"merged {cnt} times")
    return rs

def iter_bytes(b):
    for i in range(len(b)):
        yield b[i:i+1]

def train_bpe(input_path, vocab_size, special_tokens):
    buffer = read_to_buffer(input_path)
    split = split_at_special_tokens(buffer, special_tokens)
    split_as_bytes = [ i.encode() for i in split ]
    pretoken_counter = Counter(pre_tokenization(split_as_bytes))
    #print(f"{len(pretoken_counter)=}")
    token_list = [ [ b for b in iter_bytes(token) ] for token in pretoken_counter.keys() ]
    #print(f"After pre_tokenize(), {len(token_list)=}")
    vocab = build_initial_vocab(token_list, special_tokens)
    #print(f"After build_initial_vocab(), {len(vocab)=}")
    merges = []
    #print(f"{vocab_size=}, {len(vocab)=}")
    while len(vocab) < vocab_size:
        pairs = build_pairs(token_list, pretoken_counter)
        #print(f"{len(pairs)=}")
        pair = max(pairs.items(), key = lambda p: (p[1], p[0]))
        merges.append((pair[0][0], pair[0][1]))
        #print(f"merging {pair[0]=}, weight {pair[1]}")
        token_list = merge(token_list, pair[0])
        vocab[len(vocab)] = b''.join(pair[0])
        #print(f"{len(vocab)=}")
    # pre_tokenize(buf)
    # get_initial_vocab()
    return (vocab, merges)
