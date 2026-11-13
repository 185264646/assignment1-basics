#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Yang Xiwen
import regex
import itertools
from collections import Counter
from typing import List, Tuple, Dict, Set, Iterator
from dataclasses import dataclass, field

def read_to_buffer(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def split_at_special_tokens(_input: str, special_tokens: List[str]) -> List[str]:
    cur = [ _input ]
    for token in special_tokens:
        cur = [ i for i in cur for i in i.split(token) ]
    return cur

def pre_tokenization(_input: List[bytes]) -> Iterator[bytes]:
    GPT_PATTERN = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for i in _input:
        yield from regex.findall(GPT_PATTERN, i)

def build_initial_vocab(_input: List[List[bytes]], special_tokens: List[str]) -> Dict[int, bytes]:
    init = { idx: bytes([idx]) for idx in range(256) }
    init.update({ idx + 256: val.encode() for idx, val in enumerate(special_tokens) })
    return init

@dataclass
class PairState:
    weighted_count: int = 0
    affected_token_indices: List[int] = field(default_factory=list)
    last_seen_token_index: int = -1

Pairs = Dict[Tuple[bytes, bytes], PairState]

def build_init_pairs(token_list: List[List[bytes]], count_list: List[int]) -> Pairs:
    pairs: Pairs = {}
    for idx, token in enumerate(token_list):
        addition = count_list[idx]
        for pair in itertools.pairwise(token):
            pair_state = pairs.get(pair)
            if pair_state is None:
                new = PairState()
                new.weighted_count = addition
                new.affected_token_indices = [idx]
                new.last_seen_token_index = idx
                pairs[pair] = new
            else:
                pair_state.weighted_count += addition
                if pair_state.last_seen_token_index != idx:
                    pair_state.last_seen_token_index = idx
                    pair_state.affected_token_indices.append(idx)

    return pairs

def update_pairs_naive(token_list: List[List[bytes]], count_list: List[int], old_pairs: Pairs, merge_pair: Tuple[bytes, bytes]) -> None:
    pairs = build_init_pairs(token_list, count_list)
    old_pairs.clear()
    old_pairs.update(pairs)

def update_pairs(token_list: List[List[bytes]], count_list: List[int], old_pairs: Pairs, merge_pair: Tuple[bytes, bytes]) -> None:
    merged_pair = merge_pair[0] + merge_pair[1]
    recalc_pairs = set()

    for idx in old_pairs[merge_pair].affected_token_indices:
        token = token_list[idx]
        addition = count_list[idx]

        for p in itertools.pairwise(token):
            if merged_pair in p:
                # p A B q => p AB q, recalc(pA, Bq)
                # p A B A B q => p AB AB q, recalc(pA, BA, Aq)
                # p A A q => p AA q, recalc(pA, Aq)
                # p A A A q => p AA A q, recalc(pA, AA|A, Aq)
                # p A A A A q => p AA AA q, recalc(pA, AA|AA, Aq)

                # pAB => pA
                # ABq => Bq
                # ABAB => BA
                # pAA => pA
                # AAq => Aq
                # AAA => AA|A
                # AAAA => AA|AA
                if merge_pair[0] == merge_pair[1] and p[0] == merged_pair and (p[1] == merge_pair[0] or p[1] == p[0]):
                        recalc_pairs.add((p[0], p[1]))
                elif p[1] == merged_pair:
                    if p[0] == p[1]:
                        recalc_pairs.add((merge_pair[1], merge_pair[0]))
                    else:
                        recalc_pairs.add((p[0], merge_pair[0]))
                else:
                    recalc_pairs.add((merge_pair[1], p[1]))
                state = old_pairs.get(p)
                if state is None:
                    state = PairState(addition, [idx], idx)
                    old_pairs[p] = state
                else:
                    state.weighted_count += addition
                    if state.last_seen_token_index != idx:
                        state.last_seen_token_index = idx
                        state.affected_token_indices.append(idx)

    del old_pairs[merge_pair]
    affected_token_indices: Set[int] = set()
    for pair in recalc_pairs:
        affected_token_indices.update(old_pairs[pair].affected_token_indices)

    update_state_dict = { k: PairState() for k in recalc_pairs }
    for idx in sorted(affected_token_indices):
        token = token_list[idx]

        for p in itertools.pairwise(token):
            addition = count_list[idx]
            state = update_state_dict.get(p)
            if state:
                state.weighted_count += addition
                if state.last_seen_token_index != idx:
                    state.last_seen_token_index = idx
                    state.affected_token_indices.append(idx)

    old_pairs.update(update_state_dict)


def merge(lst: List[List[bytes]], pair: Tuple[bytes, bytes], update_list: List[int]) -> List[List[bytes]]:
    skip = False
    for token in (lst[i] for i in update_list):
        tmp = []
        for first, second in itertools.pairwise(itertools.chain(token, [b''])):
            if skip:
                skip = False
                continue
            if (first, second) == pair:
                tmp.append(pair[0] + pair[1])
                skip = True
            else:
                tmp.append(first)
        token[:] = tmp
    return lst

def iter_bytes(b: bytes) -> Iterator[bytes]:
    for i in range(len(b)):
        yield b[i:i+1]

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    buffer = read_to_buffer(input_path)
    split = split_at_special_tokens(buffer, special_tokens)
    split_as_bytes = [ i.encode() for i in split ]
    pretoken_counter = Counter(pre_tokenization(split_as_bytes))
    #print(f"{len(pretoken_counter)=}")
    token_list = [ [ b for b in iter_bytes(token) ] for token in pretoken_counter.keys() ]
    count_list = [ pretoken_counter[b''.join(token)] for token in token_list ]
    vocab = build_initial_vocab(token_list, special_tokens)
    pairs = build_init_pairs(token_list, count_list)
    merges = []
    while len(vocab) < vocab_size:
        pair = max(pairs.items(), key = lambda p: (p[1].weighted_count, p[0]))
        merges.append((pair[0][0], pair[0][1]))
        token_list = merge(token_list, pair[0], pair[1].affected_token_indices)
        vocab[len(vocab)] = b''.join(pair[0])
        update_pairs(token_list, count_list, pairs, pair[0])

    return (vocab, merges)
