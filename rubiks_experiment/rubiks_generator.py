from __future__ import annotations

# %%
import tokenize
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Set

import numpy as np
import torch as t

from einops import rearrange, repeat

from transformers import GPT2Tokenizer
from tokenizers import AddedToken, models, Tokenizer
from torch.utils.data import DataLoader, Dataset


cube_colors = "⬜🟨🟩🟦🟥🟧"
# Warning: the "U " token e.g. has a space at the end (so all move tokens are the same width.  Which may be dumb but whatever)
_base_moves = "UDLRFB"

all_moves = []
for m in _base_moves:
    all_moves.append(f"{m}")
    all_moves.append(f"{m}'")
    all_moves.append(f"{m}2")

non_double_moves = [m for m in all_moves if "2" not in m]

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

special_token = "<|special_token|>"
vocab = [special_token, *cube_colors, *all_moves, *all_sticker_positions_tokens]

tokenizer = GPT2Tokenizer(
    "tokenizers/vocab.json", "tokenizers/merges.txt", add_bos_token=True
)
tokenizer.add_tokens(vocab)
tokenizer.add_special_tokens({k: special_token for k in tokenizer.special_tokens_map})

device = "cuda"

color_ids = tokenizer.encode(
    cube_colors, add_special_tokens=False, return_tensors="pt"
).to(device)
move_ids = tokenizer.encode(
    all_moves, is_pretokenized=True, add_special_tokens=False, return_tensors="pt"
).to(device)
sticker_position_ids = tokenizer.encode(
    all_sticker_positions_tokens,
    is_pretokenized=True,
    add_special_tokens=False,
    return_tensors="pt",
).to(device)

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


# %%


class CubePuzzle111(CubePuzzle):
    # List of sticker values, in order (up, down, left, right, front, back)
    sticker_values: tuple

    rotation_names = [s + " " for s in "U D L R F B".split()]
    # Permutations corresponding to each rotation.
    # Rotations are named by the face they rotate clockwise.
    # e.g. U takes u -> u, d -> d, l -> b, r -> f, f -> l, b -> r
    rotations = {
        "U ": [0, 1, 4, 5, 3, 2],
        "D ": [0, 1, 5, 4, 2, 3],
        "L ": [5, 4, 2, 3, 0, 1],
        "R ": [4, 5, 2, 3, 1, 0],
        "F ": [2, 3, 1, 0, 4, 5],
        "B ": [3, 2, 0, 1, 4, 5],
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

    def observations(self):
        look_indexes = [0]
        return [sticker_to_emoji(self.sticker_values[i]) for i in look_indexes]

    def show(self):
        s = self.sticker_values
        print(f" {s[0]}")
        print(f"{s[2]}{s[4]}{s[3]}  {s[5]}")
        print(f" {s[1]}")
        print()


# %%


class CubieRepresentation(CubePuzzle):
    def __init__(self):
        # Cubie locations are ordered triples where each coordinate is -1 or 1.
        # +X is Front, +Y is Right, +Z is Up.
        # The cubie at index i in cubie_locations has stickers 3i to 3i+2.
        # positions 1, 3, 5, 7 are on the top face.
        self.cubie_locations = np.array(list(np.ndindex(2, 2, 2))) * 2 - 1

        # (0, 1, 2) means no rotation
        # (1, 2, 0) means x plane has original y sticker, y plane has original z sticker, z plane has original x sticker
        self.cubie_rotations = repeat(np.arange(3), "sticker -> cubie sticker", cubie=8)

        # fixed list of sticker values; colors are 0-5, with 2x and 2x+1 being opposing colors
        self.sticker_colors = np.zeros(24, dtype=np.int32)
        for i in range(8):
            for axis in range(3):
                self.sticker_colors[3 * i + axis] = axis * 2 + (
                    self.cubie_locations[i][axis] == -1
                )
        # print(self.sticker_colors)

    @classmethod
    def to_axis_face_direction(cls, rotation: str):
        assert 1 <= len(rotation) <= 2
        rotation_face_name = rotation[0]
        direction_name = rotation[1] if len(rotation) == 2 else " "
        axis = "FBRLUD".index(rotation_face_name) // 2
        face = "FBRLUD".index(rotation_face_name) % 2 * (-2) + 1
        direction = 1 if direction_name == " " else -1
        return axis, face, direction

    def after_move(self, rotation_name: str):
        return self.after_rotation(
            *CubieRepresentation.to_axis_face_direction(rotation_name)
        )

    def after_rotation(self, axis, face, direction) -> "CubieRepresentation":
        """
        axis is 0, 1, or 2
        face is -1 or 1
        direction is 1 for clockwise, -1 for counterclockwise
        """
        # fix cubie positions
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[axis, axis] = 1
        rotation_matrix[(axis + 1) % 3, (axis + 2) % 3] = direction * face
        rotation_matrix[(axis + 2) % 3, (axis + 1) % 3] = -direction * face
        cube = CubieRepresentation()

        cube.cubie_locations = self.cubie_locations.copy()
        for i in range(len(self.cubie_locations)):
            if self.cubie_locations[i][axis] == face:
                cube.cubie_locations[i] = rotation_matrix @ self.cubie_locations[i]

        # fix cubie rotations
        cube.cubie_rotations = self.cubie_rotations.copy()
        for i in range(len(self.cubie_rotations)):
            if cube.cubie_locations[i, axis] == face:
                # when rotating a face, the two stickers NOT on that face swap
                (
                    cube.cubie_rotations[i, (axis + 1) % 3],
                    cube.cubie_rotations[i, (axis + 2) % 3],
                ) = (
                    self.cubie_rotations[i, (axis + 2) % 3],
                    self.cubie_rotations[i, (axis + 1) % 3],
                )
        return cube

    def apply_rotation(self, axis, face, direction) -> None:
        cube = self.after_rotation(axis=axis, face=face, direction=direction)
        self.cubie_locations = cube.cubie_locations
        self.cubie_rotations = cube.cubie_rotations

    def corner_sticker_position_token_to_tuple(
        self, sticker_token: str
    ) -> Tuple[int, int, int, int]:
        """

        >>> corner_sticker_position_token_to_tuple("++-z")
        (1, 1, -1, 2)
        """
        decode_dict = {"+": 1, "-": -1, "x": 0, "y": 1, "z": 2}
        if any(c not in decode_dict for c in sticker_token):
            raise ValueError(f"Invalid sticker token {sticker_token}")
        return tuple(decode_dict[c] for c in sticker_token)

    def color_of_sticker_position(self, sticker_token: str) -> str:
        """Returns the emoji corresponding to the color of the sticker"""

        color_index = self.color_index_of_sticker_at(
            *self.corner_sticker_position_token_to_tuple(sticker_token)
        )
        return cube_colors[color_index]

    def get_position_of_sticker_id(self, sticker_id: int) -> Tuple:
        # returns (x, y, z, axis)
        cubie_id = sticker_id // 3
        xyz = self.cubie_locations[cubie_id]
        axis = list(self.cubie_rotations[cubie_id]).index(sticker_id % 3)
        return *list(xyz), axis

    def get_cubie_id_of_piece_at(self, x, y, z):
        for a in [x, y, z]:
            if a not in [-1, 1]:
                raise ValueError(f"Invalid coordinate {x, y, z}")
        matches = np.where((self.cubie_locations == [x, y, z]).all(axis=-1))[0]
        # if matches.size != 1:
        #     print(self.cubie_locations)
        #     raise ValueError(f'Expected cubie {x,y,z} to have one match, but found: {matches}')
        cubie_id = matches.item()
        assert 0 <= cubie_id < 8
        return cubie_id

    def observations(self) -> List[str]:
        """
        Returns colors as color box emoji
        """
        ret = [None] * 4
        # Gets sticker values of up (z=1) face, in order
        # 0 for (-1, -1, 1); 1 for (-1, 1, 1) etc.
        for i in range(len(ret)):
            ret[i] = cube_colors[
                self.color_index_of_sticker_at(
                    2 * (i // 2) - 1, 2 * (i % 2) - 1, 1, axis=2
                )
            ]
        return ret

    def color_index_of_sticker_at(self, x, y, z, axis):
        cubie_id = self.get_cubie_id_of_piece_at(x, y, z)
        original_sticker_axis = self.cubie_rotations[cubie_id][axis]
        # print(f"{original_sticker_axis=}", end=" ")
        ret = self.sticker_colors[cubie_id * 3 + original_sticker_axis]
        # print(f"{cubie_id=} at {x, y, z=} has color {colors[ret]} on {'xyz'[axis]}")
        return ret

    def positions_to_int(self) -> np.ndarray:
        # Returns an 8-element list of this cube's cubie positions.
        locations_binary = (self.cubie_locations + 1) // 2
        return locations_binary @ np.array([4, 2, 1])

    def positions_rotations_to_int(self) -> np.ndarray:
        # Returns an 8-element list of cubie positions and rotations, encoded between 0 and 23.
        positions = self.positions_to_int()
        rotations = self.cubie_rotations[:, 0]
        return positions * 3 + rotations

    def inverse_positions_to_int(self) -> np.ndarray:
        # Returns an 8-element list of the cubie ID at each position.
        positions = self.positions_to_int()
        return np.argsort(positions)

    def inverse_positions_rotations_to_int(self) -> np.ndarray:
        # Returns an 8-element list of cubie IDs and rotations at each position, encoded between 0 and 23.
        inverse_positions = self.inverse_positions_to_int()
        rotations = self.cubie_rotations[:, 0]
        return inverse_positions * 3 + rotations

    def sticker_colors_to_int(self) -> np.ndarray:
        # Returns a 24-element list of this cube's sticker colors, one for each position.
        ret = []
        for x in range(-1, 2, 2):
            for y in range(-1, 2, 2):
                for z in range(-1, 2, 2):
                    for axis in range(3):
                        ret.append(self.color_index_of_sticker_at(x, y, z, axis))
        return np.array(ret)

    def show(self) -> CubieRepresentation:
        # Translate the state into colors
        def c(x, y, z, axis):
            """
            Returns the color of the sticker at the given location which is showing on the given axis
            """
            color_index = self.color_index_of_sticker_at(x, y, z, axis)
            return cube_colors[color_index]

        # Assume the standard orientation (Front face at +x, Up face at +z, Right face at +y)
        l = []
        l.append("⬛⬛ {}{}".format(c(-1, -1, +1, 2), c(-1, +1, +1, 2)))  # Up face
        l.append("⬛⬛ {}{}".format(c(+1, -1, +1, 2), c(+1, +1, +1, 2)))
        l.append(
            "{}{} {}{} {}{} {}{}".format(
                c(-1, -1, +1, 1),
                c(+1, -1, +1, 1),
                c(+1, -1, +1, 0),
                c(+1, +1, +1, 0),
                c(+1, +1, +1, 1),
                c(-1, +1, +1, 1),
                c(-1, +1, +1, 0),
                c(-1, -1, +1, 0),
            )
        )  # Left, Front, Right, Back faces
        l.append(
            "{}{} {}{} {}{} {}{}".format(
                c(-1, -1, -1, 1),
                c(+1, -1, -1, 1),
                c(+1, -1, -1, 0),
                c(+1, +1, -1, 0),
                c(+1, +1, -1, 1),
                c(-1, +1, -1, 1),
                c(-1, +1, -1, 0),
                c(-1, -1, -1, 0),
            )
        )
        l.append("⬛⬛ {}{}".format(c(+1, -1, -1, 2), c(+1, +1, -1, 2)))  # Down face
        l.append("⬛⬛ {}{}".format(c(-1, -1, -1, 2), c(-1, +1, -1, 2)))
        to_print = "\n".join(l)
        to_print = to_print.replace("  ", "   ")
        print(to_print)
        return self


# %%

CubeMove = str


def generate_222_cube_data(
    data_length: int,
    rng: np.random.Generator,
    allowed_rotations_fn: Optional[
        Callable[[List[CubieRepresentation], List[CubeMove], CubeMove], bool]
    ] = None,
) -> Tuple[list, list]:
    """
    Generates a sequence of (observations..., rotation, observations..., rotation, ...),
    where rotations are random and observations are correct.
    Also returns a list of full states of the cube.
    Starts from the starting state.
    If allowed_rotations_fn is provided, it takes:
        - list of states so far, including after the proposed move
        - list of moves so far
        - a move proposed to be next
    and returns whether the move should be allowed or rejected.
    """
    # print(f"{data_length=}")
    assert isinstance(data_length, int) and data_length % 5 == 0
    n_observations = data_length // 5
    rotations = []  # rng.choice(cls.rotation_names, size=data_length // 5)
    cube = CubieRepresentation()
    states = []
    observations = []
    for i in range(n_observations):
        for j in range(1000):
            rotation_name = rng.choice(non_double_moves)
            if allowed_rotations_fn is not None:
                good_rotation = allowed_rotations_fn(
                    states + [cube.after_move(rotation_name)], rotations, rotation_name
                )
            else:
                good_rotation = True
            if good_rotation:
                rotations.append(rotation_name)
                cube = cube.after_move(rotation_name)
                observations.append(cube.observations())
                states.append(deepcopy(cube))
                break
        else:  # infinite loop?
            raise RuntimeError("Couldn't find a good rotation")
            # TODO find some way to log states without breaking dataloader
            # states.append(cube.sticker_values)
            # observations is list of strings

    # interlace observations and rotations
    data = []
    for r, o in zip(rotations, observations):
        # print(f"{type(r)=}, {type(o)=}")
        data.append(r)
        data.extend(o)

    prepend_eos = True
    token_ids = (
        tokenizer.encode(
            data,
            is_pretokenized=True,
            add_special_tokens=(not prepend_eos),
            return_tensors="pt",
        )
        .to(t.int64)
        .to(device)
    )
    return token_ids, states


def generate_2x2x2_move_query_color_poisson(
    size: int,
    rng: np.random.Generator,
    moves=non_double_moves,
    prepend_eos=True,
) -> Tuple[list, list]:
    tokens = []
    states = []
    state = CubieRepresentation()
    move_prob = rng.random()
    while len(tokens) < size:
        if rng.random() < move_prob:
            move = rng.choice(moves)
            tokens.append(move)
            states.append(state)
            state = state.after_move(move)
        else:
            query = rng.choice(corner_sticker_positions_tokens)
            color = state.color_of_sticker_position(query)
            tokens.append(query)
            states.append(state)
            tokens.append(color)
            states.append(state)
    states = states[:size]  # might have gone over
    tokens = tokens[:size]
    token_ids = (
        tokenizer.encode(
            tokens,
            is_pretokenized=True,
            add_special_tokens=(not prepend_eos),
            return_tensors="pt",
        )
        .to(t.int64)
        .to(device)
    )
    assert len(token_ids) == len(states) + (1 if prepend_eos else 0)
    return token_ids, states


def make_dataloader(
    func: Callable, batch_size: int, seq_length, num_workers=4, seed=None
) -> DataLoader:
    def my_collate(batch):
        data = t.stack([item[0] for item in batch])
        cube_state = [item[1] for item in batch]
        return data, cube_state

    dataset = FunctionalDataset(func, shape=seq_length, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=my_collate,
        num_workers=num_workers,
    )


class FunctionalDataset(Dataset):
    """
    Makes a dataset from a function that takes a shape and a random number generator.
    """

    def __init__(self, func: Callable, shape: int, seed: Optional[int] = None):
        self.shape = shape
        self.func = func
        self.seed = seed

    def __len__(self):
        return (2**31) - 1

    def __getitem__(self, idx) -> Tuple[t.Tensor, list]:
        seed = idx + 6327623784329 * self.seed if self.seed else None
        rng = np.random.default_rng(seed=seed)
        return self.func(self.shape, rng)


# %%
def __dry_test_cube(cubeclass: type) -> None:
    print(f"Dry running {cubeclass}")
    cube = cubeclass()
    # cube.show()
    # for rotation in "UDLRFB":
    #     print(f"{rotation=}")
    #     cube.apply_rotation(rotation)
    #     cube.show()

    # data, states = cubeclass.generate_random_data(10, seed=0)
    # print(f"{cubeclass.tokenize(data)=}")
    # ds = cubeclass.dataset(11, seed=0)
    # print(f"{ds[0]=}")
    dl = make_dataloader(generate_222_cube_data, batch_size=50, seq_length=125)
    for i, (data, states) in enumerate(dl):
        for pos in range(25):
            state = states[0][pos]
            datum = data[0][5 * pos + 1 : 5 * pos + 6]  # uses observations
            datum = tokenizer.decode(list(datum))
            print(f"{i=} {datum=}")
            print(f"{state.observations()=}")
            state.show()  # uses color_index_of_sticker_at
        break


if __name__ == "__main__":
    pass
    # __dry_test_cube(CubePuzzle222)
    # __dry_test_cube(CubePuzzle111)
    __dry_test_cube(CubieRepresentation)

    # cube = CubieRepresentation()
    # dl = cube.dataloader(31, batch_size=32, seed=0)
    # for i, (data, state) in enumerate(dl):    #     print(f"{i=} {data=}")
    #     break

# %%

# dl = CubieRepresentation.dataloader(11, batch_size=3, seed=0)
# for i, (data, states) in enumerate(dl):
#     print(f"{i=} {data=}")
#     print(f"{states=}")
#     my_state = states[0]
#     print(f"There are {len(states)} sequences in the batch, with {len(my_state)} states each.")
#     break

# my_state[0].show()
# my_state[1].show()

# %%
