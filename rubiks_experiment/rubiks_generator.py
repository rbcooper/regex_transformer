# %%
import tokenize
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as t

from tokenizers import AddedToken, models, Tokenizer
from torch.utils.data import DataLoader, Dataset

import rich.style


face_to_square = {
    "U": "⬜",  # White Square for Up face (white)
    "R": "🟦",  # Blue Square for Right face (blue)
    "F": "🟥",  # Red Square for Front face (red)
    "D": "🟨",  # Yellow Square for Down face (yellow)
    "L": "🟩",  # Green Square for Left face (green)
    "B": "🟧",  # Orange Square for Back face (orange)
}

_square_to_english_color_name = {
    "⬜": "white",
    "🟦": "blue",
    "🟥": "red",
    "🟨": "yellow",
    "🟩": "green",
#    "🟧": "orange",
    "🟧": "bright_red",
}

identity_move2x2x2 = list(range(24))

basic_moves2x2x2 = {
    "U": [
        0,
        1,
        12,
        13,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        14,
        15,
        18,
        19,
        16,
        17,
        20,
        21,
        22,
        23,
    ],
    "D": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        10,
        11,
        8,
        9,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        22,
        23,
        20,
        21,
    ],
    "L": [
        20,
        21,
        2,
        3,
        0,
        1,
        6,
        7,
        4,
        5,
        10,
        11,
        12,
        13,
        14,
        15,
        8,
        17,
        18,
        19,
        16,
        9,
        22,
        23,
    ],
    "R": [
        0,
        1,
        6,
        7,
        4,
        5,
        22,
        23,
        8,
        9,
        20,
        21,
        2,
        3,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        12,
        13,
    ],
    "F": [
        16,
        17,
        18,
        19,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        20,
        21,
        22,
        23,
        0,
        1,
        2,
        3,
    ],
    "B": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        22,
        23,
        20,
        21,
        16,
        17,
        18,
        19,
        12,
        13,
        14,
        15,
        8,
        9,
        10,
        11,
    ],
}


def _debug_moves():
    for label, move in basic_moves2x2x2.items():
        to_the_forth = permute_product([move] * 4)
        if to_the_forth != identity_move2x2x2:
            print(f"Move {label} is not of order 4: {to_the_forth}")
            _debug_basic_move(move)


def _debug_basic_move(move):
    for i in range(5):
        print(f"^{i}")
        pretty_print_cube_state(permute_product([move] * i), as_number=True)


def multiply_permutations(perm1, perm2):
    if len(perm1) != len(perm2):
        raise ValueError("Permutations must have the same length")

    n = len(perm1)
    result = [0] * n

    for i in range(n):
        result[i] = perm2[perm1[i]]

    return result


def permute_product(
    permutations: Sequence[Sequence[int]], identity=identity_move2x2x2
) -> Sequence[int]:
    result = identity
    for perm in permutations:
        result = multiply_permutations(result, perm)
    return result


for label, move in basic_moves2x2x2.items():
    to_the_forth = permute_product([move] * 4)
    if to_the_forth != identity_move2x2x2:
        print(f"Move {label} is not of order 4: {to_the_forth}")


def generate_sticker_to_color():
    face_colors = ["U", "R", "F", "D", "L", "B"]
    stickers_per_face = 4
    sticker_to_color = {}

    for i in range(len(face_colors)):
        for j in range(stickers_per_face):
            sticker_to_color[i * stickers_per_face + j] = face_colors[i]

    return sticker_to_color


sticker_to_color = generate_sticker_to_color()


def sticker_to_emoji(s : Union[int, str], as_number:bool = False) -> str:
    if as_number:
        if not isinstance(s, int):
            raise ValueError(f"Need an int to display sticker as int {type(s)=}")
        color_name = _square_to_english_color_name[sticker_to_emoji(s)]
        return rich.style.Style(color=color_name).render(f"{s:2} ")

    if s in face_to_square.values():
        return s
    elif s in face_to_square.keys():
        return face_to_square[s]
    elif s in identity_move2x2x2:
        return sticker_to_emoji(sticker_to_color[s])
    else:
        raise ValueError(f"Invalid state: {s}")


def pretty_print_cube_state(state: Sequence, do_print=True, as_number=False) -> str:
    # Translate the state into colors
    stickers = [sticker_to_emoji(s, as_number=as_number) for s in state]
    if len(stickers) != 24:
        raise ValueError(f"Bad state length: {len(state)=} {state=}")

    # Assume the standard orientation (Front face at front, Up face at top)
    l = []
    l.append("      {}{}".format(stickers[16], stickers[17]))  # Up face
    l.append("      {}{}".format(stickers[18], stickers[19]))
    l.append(
        "{}{} {}{} {}{} {}{}".format(
            stickers[12],
            stickers[13],
            stickers[0],
            stickers[1],
            stickers[4],
            stickers[5],
            stickers[8],
            stickers[9],
        )
    )  # Left, Front, Right, Back faces
    l.append(
        "{}{} {}{} {}{} {}{}".format(
            stickers[14],
            stickers[15],
            stickers[2],
            stickers[3],
            stickers[6],
            stickers[7],
            stickers[10],
            stickers[11],
        )
    )
    l.append("      {}{}".format(stickers[20], stickers[21]))  # Down face
    l.append("      {}{}".format(stickers[22], stickers[23]))
    to_print = "\n".join(l)
    to_print = to_print.replace("  ", "   ")
    if do_print:
        print(to_print)


# %%

pretty_print_cube_state(identity_move2x2x2)

# %%


def multiply_permutations(perm1, perm2):
    if len(perm1) != len(perm2):
        raise ValueError("Permutations must have the same length")

    n = len(perm1)
    result = [0] * n

    for i in range(n):
        result[i] = perm2[perm1[i]]

    return result


def permute_product(
    permutations: Sequence[Sequence[int]], identity=identity_move2x2x2
) -> Sequence[int]:
    result = identity
    for perm in permutations:
        result = multiply_permutations(result, perm)
    return result


all_moves2x2x2 = []
for move in basic_moves2x2x2:
    all_moves2x2x2.append(move)
    all_moves2x2x2.append(move + "'")
    all_moves2x2x2.append(move + "2")

vocab = [*face_to_square.values(), *all_moves2x2x2]

tokenizer = Tokenizer(models.Unigram())

for word in vocab:
    tokenizer.add_tokens([AddedToken(word, normalized=False)])
# %%


# %%
class CubePuzzle:
    def __init__():
        pass

    def sticker_value(self, face, row, col):
        pass

    def cubie_position(self, cubie_id: int):
        pass

    def apply_rotation(self, face, direction):
        pass

    @staticmethod
    def generate_random_data(self, data_length: int) -> Tuple[list, list]:
        """
        For the 2x2x2 cube, generates a sequence of (4x sticker observation, rotation) pairs,
        where rotations are random and observations are correct.
        Also returns a list of full states of the cube.
        Starts from the starting state.
        e.g. ([3, 1, 4, 1, 10, ...], [CubePuzzle(...), CubePuzzle(...), ...])
        """
        moves = "U D L R F B".split()
        pass


# %%


class CubePuzzle111(CubePuzzle):
    # List of sticker values, in order (up, down, left, right, front, back)
    sticker_values: tuple

    rotation_names = "U D L R F B".split()
    # Permutations corresponding to each rotation.
    # Rotations are named by the face they rotate clockwise.
    # e.g. U takes u -> u, d -> d, l -> b, r -> f, f -> l, b -> r
    rotations = {
        "U": [0, 1, 4, 5, 3, 2],
        "D": [0, 1, 5, 4, 2, 3],
        "L": [5, 4, 2, 3, 0, 1],
        "R": [4, 5, 2, 3, 1, 0],
        "F": [2, 3, 1, 0, 4, 5],
        "B": [3, 2, 0, 1, 4, 5],
    }

    def __init__(self):
        self.sticker_values = list(range(6))

    def sticker_value(self, face, row, col):
        return self.sticker_values[face]

    def cubie_position(self, cubie_id: int):
        return NotImplementedError

    def apply_rotation(self, rotation_name: str):
        """
        For the 1x1x1 cube, there are only 6 rotations.
        """
        # When applying a rotation, the NEW ith sticker value is the OLD rotation[i]'th sticker value.
        self.sticker_values = tuple(
            self.sticker_values[i] for i in self.rotations[rotation_name]
        )

    def copy(self):
        new_cube = CubePuzzle111()
        new_cube.sticker_values = self.sticker_values
        return new_cube

    def __repr__(self):
        return f"CubePuzzle111({self.sticker_values})"

    def show(self):
        s = self.sticker_values
        print(f" {s[0]}")
        print(f"{s[2]}{s[4]}{s[3]}  {s[5]}")
        print(f" {s[1]}")
        print()

    @classmethod
    def dataset(cls, data_length: int, seed=None) -> Dataset:
        """A torch dataset yielding (random_word, dfa_state) pairs)"""
        return CubePuzzleDataset(cls, data_length, seed=seed)

    @classmethod
    def dataloader(cls, data_length: int, batch_size: int, seed=None) -> DataLoader:
        return DataLoader(
            CubePuzzle111.dataset(cls, data_length, seed=seed), batch_size=batch_size
        )

    @classmethod
    def generate_random_data(self, data_length: int, seed: int) -> Tuple[list, list]:
        """
        For the 1x1x1 cube, generates a sequence of (observation, rotation, observation, rotation, ...),
        where rotations are random and observations are correct.
        Also returns a list of full states of the cube.
        Starts from the starting state.
        """
        assert isinstance(data_length, int) and data_length % 2 == 1
        rng = np.random.default_rng(seed)
        rotations = rng.choice(self.rotation_names, size=data_length // 2)
        cube = CubePuzzle111()
        states = []
        observations = [cube.sticker_values[0]]
        for rotation in rotations:
            cube.apply_rotation(rotation)
            states.append(cube.sticker_values)
            observations.append(cube.sticker_values[0])

        # interlace observations and rotations
        data = [None] * (data_length)
        data[1::2] = rotations
        data[::2] = observations
        return data, states

    @classmethod
    def tokenize(self, data: list, prepend_eos=True):
        """
        Returns a tensor with interlaced observations and rotations.
        Observations are ASCII codes for numbers 0-5, rotations are letters.
        Also prepends the EOS token 0.
        """
        result = t.tensor(tokenizer.encode(data, is_pretokenized=True).ids, dtype=t.int32)
        if prepend_eos:
            result = t.cat([t.tensor([0], dtype=t.int32), result])
        return result


# %%

class CubePuzzle222(CubePuzzle):
    identity = identity_move2x2x2
    # List of sticker values, in order (up, down, left, right, front, back)
    sticker_values: List[int]

    rotation_names = "U D L R F B".split()
    # Permutations corresponding to each rotation.
    # Rotations are named by the face they rotate clockwise.
    # e.g. U takes u -> u, d -> d, l -> b, r -> f, f -> l, b -> r
    # TODO add other rotations
    rotations = basic_moves2x2x2


    def __init__(self):
        self.sticker_values = CubePuzzle222.identity


    def sticker_value(self, face, row, col):
        return self.sticker_values[face]

    def cubie_position(self, cubie_id: int):
        return NotImplementedError

    def apply_rotation(self, rotation_name:str):
        """
        For the 1x1x1 cube, there are only 6 rotations.
        """
        # When applying a rotation, the NEW ith sticker value is the OLD rotation[i]'th sticker value.
        r = self.rotations[rotation_name]
        result = multiply_permutations(self.sticker_values, r)
        self.sticker_values = result

    def copy(self):
        new_cube = CubePuzzle222()
        new_cube.sticker_values = self.sticker_values
        return new_cube

    def __repr__(self):
        return f"CubePuzzle222({self.sticker_values})"

    def show(self):
        pretty_print_cube_state(self.sticker_values, do_print=True)

    def observations(self):
        look_indexes = [0, 1, 2, 3]
        return [sticker_to_emoji(self.sticker_values[i]) for i in look_indexes]

    @classmethod
    def dataset(cls, data_length: int, seed=None) -> Dataset:
        """A torch dataset yielding (random_word, dfa_state) pairs)
        """
        return CubePuzzleDataset(cls, data_length=data_length, seed=seed)

    @classmethod
    def dataloader(cls, data_length: int, batch_size: int, seed=None) -> DataLoader:
        return DataLoader(cls.dataset(data_length, seed=seed), batch_size=batch_size)

    @classmethod
    def generate_random_data(cls, data_length:int, seed:int) -> Tuple[list, list]:
        """
        For the 1x1x1 cube, generates a sequence of (observation, rotation, observation, rotation, ...),
        where rotations are random and observations are correct.
        Also returns a list of full states of the cube.
        Starts from the starting state.
        """
        assert isinstance(data_length, int) and data_length % 2 == 1
        rng = np.random.default_rng(seed)
        rotations = rng.choice(cls.rotation_names, size=data_length // 2)
        cube = cls()
        states = []
        observations = []
        for rotation in rotations:
            cube.apply_rotation(rotation)
            states.append(cube.sticker_values)
            observations.append(cube.observations())

        # interlace observations and rotations
        data = []
        for r, o in zip(rotations, observations):
            data.append(r)
            data.extend(o)
        return data, states

    @classmethod
    def tokenize(self, data: list, prepend_eos=True):
        """
        Returns a tensor with interlaced observations and rotations.
        Observations are ASCII codes for numbers 0-5, rotations are letters.
        Also prepends the EOS token 0.
        """
        result = t.tensor(tokenizer.encode(data, is_pretokenized=True).ids, dtype=t.int32)
        if prepend_eos:
            result = t.cat([t.tensor([0], dtype=t.int32), result])
        return result


class CubePuzzleDataset(Dataset):
    def __init__(
        self,
        group_class: type,
        data_length: int,
        *,
        seed: Optional[int] = None,
        prepend_eos: bool = True,
    ):
        self.group_class = group_class
        self.data_length = data_length
        self.seed = seed
        self.prepend_eos = prepend_eos

    def __len__(self):
        return (2**31) - 1

    def __getitem__(self, idx):
        data, states = self.group_class.generate_random_data(
            self.data_length - self.prepend_eos, seed=idx
        )
        data = self.group_class.tokenize(data, prepend_eos=self.prepend_eos)
        return data, states

# %%
_debug_moves()
# %%

if __name__ == "__main__":
    cube = CubePuzzle222()
    cube.show()
    for rotation in "UDLRFB":
        print(f"{rotation=}")
        cube.apply_rotation(rotation)
        cube.show()

    data, states = CubePuzzle222.generate_random_data(11, seed=0)
    print(f"{CubePuzzle222.tokenize(data)=}")
    ds = CubePuzzle222.dataset(10, seed=0)
    print(f"{ds[0]=}")
# %%
