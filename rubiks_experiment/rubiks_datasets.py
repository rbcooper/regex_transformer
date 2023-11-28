# %%
from collections import Counter
from typing import Callable, Generator, List, Optional, Tuple, Union
import numpy as np
import pycuber

from jaxtyping import Float, Int

from transformers import GPT2Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

import torch
from torch.utils.data import Dataset


# %%


"""String of color emojis for faces of the cube"""
cube_colors = "⬜🟨🟩🟦🟥🟧"

cube_colors_english = ["white", "yellow", "green", "blue", "red", "orange"]
english_to_token = {c: t for c, t in zip(cube_colors_english, cube_colors)}
# Warning: the "U " token e.g. has a space at the end (so all move tokens are the same width.  Which may be dumb but whatever)
_base_moves = "UDLRFB"

all_moves = []
for m in _base_moves:
    all_moves.append(f"{m}")
    all_moves.append(f"{m}'")
    all_moves.append(f"{m}2")

non_double_moves = [m for m in all_moves if "2" not in m]

"""All strings that represent a position of a sticker on the cube.
+-+y means:
    the x axis is on the positive side (right)
    the y axis is on the negative side (towards the viewer)
    the z axis in on the positive side (up)
    the sticker is on the y axis (behind face)
"""
all_sticker_positions_tokens = []
for x in "-0+":
    for y in "-0+":
        for z in "-0+":
            for i, axis in enumerate("xyz"):
                xyz = f"{x}{y}{z}"
                if xyz[i] not in "+-":
                    # Can't have a sticker that isn't on the edge
                    continue
                all_sticker_positions_tokens.append(f"{x}{y}{z}{axis}")

corner_sticker_positions_tokens = [
    t for t in all_sticker_positions_tokens if "0" not in t
]

_special_token = "<|special_token|>"
special_tokens = [_special_token]
vocab = [*special_tokens, *cube_colors, *all_moves, *all_sticker_positions_tokens]

"""
Tokenizer for rubiks cube moves, colors, and sticker positions.

>>> tokenizer("⬜ U U' L2 -+0y DDD")['input_ids']
[0, 1, 7, 8, 15, 42, 10, 10, 10]
"""
tokenizer: PreTrainedTokenizer = GPT2Tokenizer(
    "tokenizers/vocab.json", "tokenizers/merges.txt", add_bos_token=True
)
tokenizer.add_tokens(vocab)
tokenizer.add_special_tokens({k: _special_token for k in tokenizer.special_tokens_map})
tokenizer.pad_token = tokenizer.eos_token


def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (str) using the vocab."""
    return self.decoder.get(index, self.unk_token)


# Fix bug in GPT2Tokenizer when there are more tokens in the model than in the vocab
GPT2Tokenizer._convert_id_to_token = _convert_id_to_token

device = "cuda"

"""Tokenizer ids of the colors of the cube"""
color_ids = tokenizer.encode(
    cube_colors, add_special_tokens=False, return_tensors="pt"
)[0]
all_move_ids = tokenizer.encode(
    all_moves, is_pretokenized=True, add_special_tokens=False, return_tensors="pt"
)[0]
all_sticker_position_ids = tokenizer.encode(
    all_sticker_positions_tokens,
    is_pretokenized=True,
    add_special_tokens=False,
    return_tensors="pt",
)[0]
corner_sticker_position_ids = tokenizer.encode(
    corner_sticker_positions_tokens,
    is_pretokenized=True,
    add_special_tokens=False,
    return_tensors="pt",
)[0]

special_tokens_ids = tokenizer.encode(
    special_tokens, is_pretokenized=True, add_special_tokens=False, return_tensors="pt"
)[0]

all_tokens_ids = tokenizer.encode(
    vocab, is_pretokenized=True, add_special_tokens=False, return_tensors="pt"
)[0]


def ensure_token_list(tokens):
    """
    Converts a string, list of strings or pytorch tensor into a list of token strings.
    """
    result = []
    if isinstance(tokens, list):
        if all(isinstance(w, int) for w in tokens):
            result = tokenizer.convert_ids_to_tokens(tokens)
        elif all(isinstance(w, str) for w in tokens):
            if not all(w in vocab for w in tokens):
                raise ValueError(f"Not all tokens are in the vocab: {tokens}")
            result = tokens
    elif isinstance(tokens, str):
        result = tokenizer.tokenize(tokens)
    elif isinstance(tokens, torch.Tensor):
        result = tokenizer.convert_ids_to_tokens(tokens.tolist())
    for i, t in enumerate(result):
        result[i] = str(t)
    return result


# %%


_axis_to_pycube_face = {
    "x": "LR",
    "y": "BF",
    "z": "DU",
}

_pycube_face_to_axis = {}
for axis, letters in _axis_to_pycube_face.items():
    for l in letters:
        _pycube_face_to_axis[l] = axis

_query_token_to_pycube_position = {}

for qt in all_sticker_positions_tokens:
    x, y, z, r = qt
    result_parts = []
    for axis_l, (p, m) in zip([x, y, z], _axis_to_pycube_face.values()):
        l = axis_l.replace("+", p).replace("-", m).replace("0", "")
        result_parts.append(l)
    _query_token_to_pycube_position[qt] = "".join(result_parts)

# %%


def _ensure_pycube_cubie(position):
    """Ensures a string position is in pycuber format for a cubie ("LDB", "LB", etc)
    >>> _ensure_pycube_position("+++y")
    "UBR"
    """
    if len(position) in [1, 2, 3]:
        # Already right format
        if all(c in _base_moves for c in position):
            return position
    if len(position) == 4:
        if position in _query_token_to_pycube_position:
            return _query_token_to_pycube_position[position]
    raise ValueError(f"Invalid position {position}")


def _ensure_pycube_face(position: str):
    if len(position) == 4:
        axis = position[3]
        cube_faces = _ensure_pycube_cubie(position)
        possible_axis_face = _axis_to_pycube_face[axis]
        faces = [f for f in possible_axis_face if f in cube_faces]
        assert len(faces) == 1
        return faces[0]
    raise ValueError(f"Invalid position {position}")


SOLVED_CUBE = pycuber.Cube()


def color_at(cube, position):
    if isinstance(cube, pycuber.Cube):
        cubie = cube[_ensure_pycube_cubie(position)]
        face = _ensure_pycube_face(position)
        face_color = cubie[face].colour
        return english_to_token[face_color]


def zip_with_pycuber_cube(tokens) -> Generator[Tuple[str, pycuber.Cube], None, None]:
    """Yields (token, cube) pairs for all the states of the tokens"""
    tokens = ensure_token_list(tokens)
    cube = pycuber.Cube()
    for token in tokens:
        if token in all_moves:
            cube.perform_algo(token)
        yield token, cube


SingleSampleTokens = Union[str, List[str], List[int], Int[torch.Tensor, "pos"]]
DataGeneratorFunction = Callable[[int, np.random.Generator], SingleSampleTokens]


def ensure_tokenized_tensor(
    tokens: SingleSampleTokens, add_bos=False
) -> Int[torch.Tensor, "pos"]:
    return _ensure_tokenized1(tokens, add_bos=add_bos)


def _ensure_tokenized1(
    tokens: SingleSampleTokens, add_bos=False
) -> Int[torch.Tensor, "pos"]:
    """
    Does NOT add special tokens
    """
    result = None
    if isinstance(tokens, str):
        result = tokenizer.encode(tokens, add_special_tokens=add_bos)
    elif isinstance(tokens, list):
        if all(isinstance(w, str) and w in vocab for w in tokens):
            result = tokenizer.encode(tokens, add_special_tokens=add_bos)
        elif all(isinstance(w, int) and w in all_tokens_ids for w in tokens):
            result = torch.tensor(tokens)
    elif isinstance(tokens, torch.Tensor):
        if tokens.dtype in [torch.int64, torch.int32]:
            if torch.isin(tokens, all_tokens_ids).all():
                result = tokens
    if result is None:
        raise ValueError(f"Invalid tokens {tokens}")
    if add_bos:
        assert result[0] == tokenizer.bos_token_id
        if len(result) > 1:
            assert result[1] != tokenizer.bos_token_id
    return result


# TODO: _ensure_tokenized_batch


def with_correct_colors_after_queries(tokens: SingleSampleTokens, truncate=True):
    """Makes a "correct" version of tokens.  Drop the original colors"""
    length = len(tokens)
    result = []
    for token, cube in zip_with_pycuber_cube(tokens):
        if token not in vocab:
            raise ValueError(f"Token {token} is not a move")
        elif token in cube_colors:
            continue
        elif token in all_moves:
            result.append(token)
        elif token in all_sticker_positions_tokens:
            result.append(token)
            result.append(color_at(cube, token))
        elif token in special_tokens:
            result.append(token)
        else:
            assert False, "Unreachable"
    if truncate:
        result = result[:length]
    return result


def is_valid_sequence(tokens: SingleSampleTokens) -> bool:
    """Returns true if the colors of the sequence are possible, otherwise false"""
    tokens = ensure_token_list(tokens)
    valids = with_correct_colors_after_queries(tokens, truncate=True)
    valids = ensure_token_list(valids)
    result = tokens == valids
    return result


def pretty_incorrect_colors(tokens):
    tokens = ensure_token_list(tokens)
    correct = with_correct_colors_after_queries(tokens, truncate=True)
    result = []
    for a, c in zip(tokens, correct):
        if a == c:
            result.append(a)
        else:
            result.append("❌")
    return " ".join(result)


class FunctionalDataset(Dataset):
    """
    Makes a dataset from a function that takes a shape and a random number generator.
    """

    def __init__(
        self,
        func: DataGeneratorFunction,
        *,
        length: int,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.func = func
        self.length = length
        self.seed = seed
        self.name = name or func.__name__

    def __len__(self):
        return (2**31) - 1

    def __getitem__(self, idx) -> Int[torch.Tensor, "pos"]:
        debug = False
        if debug:
            print(f"{self.name} {idx}")
        seed = idx + 6327623784329 * self.seed if self.seed else None
        rng = np.random.default_rng(seed=seed)
        result = self.func(rng=rng, length=self.length)
        result = ensure_tokenized_tensor(result, add_bos=True)
        if debug:
            print(f"end {self.name} {idx}")
        return result

    def __repr__(self) -> str:
        """Returns a string representation of the dataset, using the fields length, seed, and name"""
        return f"{self.__class__.__name__}(length={self.length}, seed={self.seed}, name={self.name})"


def make_sample_with_query_prob(
    *,
    rng: np.random.Generator,
    length: int,
    query_prob: float,
    moves=non_double_moves,
    sticker_positions=corner_sticker_positions_tokens,
    add_bos=True,
) -> Int[torch.Tensor, "pos"]:
    result = []
    for _ in range(length):
        if rng.random() < query_prob:
            result.append(rng.choice(sticker_positions))
        else:
            result.append(rng.choice(moves))
    result = with_correct_colors_after_queries(result)
    result = ensure_tokenized_tensor(result, add_bos=add_bos)
    result = result[:length]
    assert len(result) == length
    return result


def make_prob_query_dataset(
    length: int, *, query_prob: float, moves=non_double_moves, seed=None
) -> Dataset:
    """Creates a dataset of random sequences of tokens with a given probability of queries"""

    def gen_sample(rng: np.random.Generator, length=length) -> Int[torch.Tensor, "pos"]:
        return make_sample_with_query_prob(
            rng=rng, length=length, query_prob=query_prob, moves=moves
        )

    return FunctionalDataset(
        gen_sample,
        length=length,
        name=f"query_prob_{query_prob:.2f}".replace(".", "_"),
        seed=seed,
    )


def make_start_cube_dataset(length: int, seed=None) -> Dataset:
    """Returns a dataset with queries and colors only.  No moves"""

    def gen_sample(rng: np.random.Generator, length=length) -> Int[torch.Tensor, "pos"]:
        return make_sample_with_query_prob(
            rng=rng, length=length, query_prob=1.0, moves=[]
        )

    return FunctionalDataset(
        gen_sample, length=length, name=f"start_cube_query_only", seed=seed
    )


def make_only_moves_dataset(length: int, moves=non_double_moves, seed=None) -> Dataset:
    """Returns a dataset with queries and colors only.  No moves"""

    def gen_sample(rng: np.random.Generator, length=length) -> Int[torch.Tensor, "pos"]:
        return make_sample_with_query_prob(
            rng=rng, length=length, query_prob=0.0, moves=moves
        )

    return FunctionalDataset(gen_sample, length=length, name=f"moves_only", seed=seed)


def make_uniform_prob_dataset(
    length: int, moves=non_double_moves, seed=None
) -> Dataset:
    """Returns a dataset with queries and colors only.  No moves"""

    def gen_sample(rng: np.random.Generator, length=length) -> Int[torch.Tensor, "pos"]:
        query_prob = rng.random()
        return make_sample_with_query_prob(
            rng=rng, length=length, query_prob=query_prob, moves=moves
        )

    return FunctionalDataset(gen_sample, length=length, name=f"uniform_prob", seed=seed)


def make_n_moves_then_query_dataset(
    length: int,
    n_moves: int,
    moves=non_double_moves,
    sticker_positions=corner_sticker_positions_tokens,
    seed=None,
) -> Dataset:
    """Returns a that does n moves, then only queries"""
    if n_moves > length:
        raise ValueError(f"n_moves {n_moves} > length {length}")

    def gen_sample(rng: np.random.Generator, length=length) -> Int[torch.Tensor, "pos"]:
        result = []
        for _ in range(n_moves):
            result.append(rng.choice(moves))
        while len(result) < length:
            result.append(rng.choice(sticker_positions))
        result = with_correct_colors_after_queries(result)
        result = ensure_tokenized_tensor(result, add_bos=True)
        result = result[:length]
        assert len(result) == length
        return result

    return FunctionalDataset(
        gen_sample, length=length, name=f"first_{n_moves}_moves_then_queries", seed=seed
    )


# %%
def print_dataset_info(dataset: Dataset, n=30):
    print(f"dataset name: {dataset.name}")
    moves = Counter()
    queries = Counter()
    colors = Counter()
    specials = Counter()
    for i in range(n):
        sample = dataset[i]
        print(tokenizer.decode(sample))
        moves.update(tokenizer.decode(t) for t in sample if t in all_move_ids)
        queries.update(
            tokenizer.decode(t) for t in sample if t in all_sticker_position_ids
        )
        colors.update(tokenizer.decode(t) for t in sample if t in color_ids)
        specials.update(tokenizer.decode(t) for t in sample if t in special_tokens_ids)
    print(f"moves: ({len(moves)}) {moves}")
    print(f"queries: ({len(queries)}) {queries}")
    print(f"color: ({len(colors)}) {colors}")


# %%
