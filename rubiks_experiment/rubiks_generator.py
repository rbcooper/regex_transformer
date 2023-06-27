# %%
import tokenize
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as t

from tokenizers import AddedToken, models, Tokenizer
from torch.utils.data import DataLoader, Dataset

from einops import rearrange, repeat

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


identity_move2x2x2 = list(range(24))


def permute_product(
    permutations: Sequence[Sequence[int]], identity=identity_move2x2x2
) -> Sequence[int]:
    result = identity
    for perm in permutations:
        result = multiply_permutations(result, perm)
    return result


def _debug_moves():
    for label, move in basic_moves2x2x2.items():
        to_the_forth = permute_product([move] * 4)
        if to_the_forth != identity_move2x2x2:
            print(f"Move {label} is not of order 4: {to_the_forth}")
            _debug_basic_move(move)


__basic_moves_strs = {
    "U": " 0  1 12 13  2  3  4  5  6  7  8  9 10 11 14 15 18 19 16 17 20 21 22 23",
    "D": " 0  1  2  3  4  5  6  7 10 11  8  9 12 13 14 15 16 17 18 19 22 23 20 21",
    "L": "20 21  2  3  0  1  6  7  4  5 10 11 12 13 14 15  8 17 18 19 16  9 22 23",
    "R": " 0  1  6  7  4  5 22 23  8  9 20 21  2  3 14 15 16 17 18 19 20 21 12 13",
    "F": "16 17 18 19  4  5  6  7  8  9 10 11 12 13 14 15 20 21 22 23  0  1  2  3",
    "B": " 0  1  2  3  4  5  6  7 22 23 20 21 16 17 18 19 12 13 14 15  8  9 10 11",
}
basic_moves2x2x2 = {
    k: [int(s) for s in v.split()] for k, v in __basic_moves_strs.items()
}


for label, move in basic_moves2x2x2.items():
    to_the_forth = permute_product([move] * 4)
    if to_the_forth != identity_move2x2x2:
        print(f"Move {label} is not of order 4: {to_the_forth}")


def _generate_sticker_to_color(stickers_per_face=4):
    face_colors = ["U", "R", "F", "D", "L", "B"]
    sticker_to_color = {}

    for i in range(len(face_colors)):
        for j in range(stickers_per_face):
            sticker_to_color[i * stickers_per_face + j] = face_colors[i]

    return sticker_to_color


sticker_to_color = _generate_sticker_to_color()


def sticker_to_emoji(s: Union[int, str], as_number: bool = False) -> str:
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

    def observations(self) -> List[str]:
        pass

    @classmethod
    def dataset(cls, data_length: int, seed=None) -> Dataset:
        """A torch dataset yielding (random_word, dfa_state) pairs)"""
        return CubePuzzleDataset(cls, data_length=data_length, seed=seed)

    @classmethod
    def dataloader(cls, data_length: int, batch_size: int, seed=None) -> DataLoader:
        def my_collate(batch):
            data = t.stack([item[0] for item in batch])
            target = [item[1] for item in batch]
            return data, target
        return DataLoader(cls.dataset(data_length, seed=seed), batch_size=batch_size, collate_fn=my_collate)

    @classmethod
    def generate_random_data(cls, data_length: int, seed: int) -> Tuple[list, list]:
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
        result = t.tensor(
            tokenizer.encode(data, is_pretokenized=True).ids, dtype=t.int64
        )
        if prepend_eos:
            result = t.cat([t.tensor([0], dtype=t.int64), result])
        return result


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

    def apply_rotation(self, rotation_name: str):
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

    def observations(self) -> List[str]:
        look_indexes = [0, 1, 2, 3]
        return [sticker_to_emoji(self.sticker_values[i]) for i in look_indexes]

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
        print(f"dataset to generate data of length {self.data_length - self.prepend_eos}")

    def __len__(self):
        return (2**31) - 1

    def __getitem__(self, idx) -> Tuple[t.Tensor, list]:
        data, states = self.group_class.generate_random_data(
            self.data_length - self.prepend_eos, seed=idx
        )
        data = self.group_class.tokenize(data, prepend_eos=self.prepend_eos)
        return data, states


# %%


class CubieRepresentation(CubePuzzle):
    colors = '⬜🟨🟦🟩🟥🟧'
    def __init__(self):
        # cubie i has stickers 3i to 3i+2
        self.cubie_locations = np.array(list(np.ndindex(2,2,2))) * 2 - 1

        # (0, 1, 2) means no rotation
        # (1, 2, 0) means x plane has original y sticker, y plane has original z sticker, z plane has original x sticker
        self.cubie_rotations = repeat(np.arange(3), 'sticker -> cubie sticker', cubie=8)

        # fixed list of sticker values; colors are 0-5, with 2x and 2x+1 being opposing colors
        self.sticker_colors = np.zeros(24, dtype=np.int32)
        for i in range(8):
            for axis in range(3):
                self.sticker_colors[3*i + axis] = axis * 2 + (self.cubie_locations[i][axis] == -1)
        # print(self.sticker_colors)

    def apply_rotation(self, rotation:str):
        raise NotImplementedError

    def apply_rotation(self, axis, face, direction):
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

        for i in range(len(self.cubie_locations)):
            if self.cubie_locations[i][axis] == face:
                self.cubie_locations[i] = rotation_matrix @ self.cubie_locations[i]
        # fix cubie rotations
        for i in range(len(self.cubie_rotations)):
            if self.cubie_locations[i, axis] == face:
                # when rotating a face, the two stickers NOT on that face swap
                self.cubie_rotations[i, (axis + 1) % 3], self.cubie_rotations[i, (axis + 2) % 3] = self.cubie_rotations[i, (axis + 2) % 3], self.cubie_rotations[i, (axis + 1) % 3]
    
    def get_cubie_id_of_piece_at(self, x, y, z):
        for a in [x, y, z]:
            if a not in [-1, 1]:
                raise ValueError(f"Invalid coordinate {x, y, z}")
        cubie_id = np.where((self.cubie_locations == [x, y, z]).all(axis=-1))[0].item()
        assert 0 <= cubie_id < 8
        return cubie_id

    def observations(self) -> List[str]:
        """
        Returns colors as color box emoji
        """
        ret = [None]*4
        # Gets sticker values of up face, in order
        for i in range(len(self.cubie_locations)):
            if self.cubie_locations[i][2] == 1:
                # this is on the top face; get the z sticker
                cubie_location = (self.cubie_locations[i][0] == 1) * 2 + (self.cubie_locations[i][1] == 1)
                z_sticker = 3*i + list(self.cubie_rotations[i]).index(2)
                ret[cubie_location] = CubieRepresentation.colors[self.sticker_colors[z_sticker]]
        return ret
    
    def color_index_of_sticker_at(self, x, y, z, axis):

        cubie_id = self.get_cubie_id_of_piece_at(x, y, z)
        original_sticker_axis = self.cubie_rotations[cubie_id][axis]
        ret = self.sticker_colors[cubie_id * 3 + original_sticker_axis]
        # print(f"{cubie_id=} at {x, y, z=} has color {colors[ret]} on {'xyz'[axis]}")
        return ret

    @classmethod
    def generate_random_data(cls, data_length: int, seed: int) -> Tuple[list, list]:
        """
        For the 1x1x1 cube, generates a sequence of (observation, rotation, observation, rotation, ...),
        where rotations are random and observations are correct.
        Also returns a list of full states of the cube.
        Starts from the starting state.
        """
        # print(f"{data_length=}")
        assert isinstance(data_length, int) and data_length % 5 == 0
        n_observations = data_length // 5
        rng = np.random.default_rng(seed)
        rotations = [] # rng.choice(cls.rotation_names, size=data_length // 5)
        cube = cls()
        states = []
        observations = []
        for i in range(n_observations):
            axis = rng.integers(3)
            face = rng.choice([-1, 1])
            direction = rng.choice([-1, 1])
            rotation_name = "U D L R F B".split()[axis * 2 + (face == -1)]
            if direction == -1:
                rotation_name += "'"
            rotations.append(rotation_name)
            cube.apply_rotation(axis, face, direction)
            # TODO find some way to log states without breaking dataloader
            # states.append(cube.sticker_values)
            # observations is list of strings
            observations.append(cube.observations())

        # interlace observations and rotations
        data = []
        for r, o in zip(rotations, observations):
            # print(f"{type(r)=}, {type(o)=}")
            data.append(r)
            data.extend(o)
        return data, None # states

    def show(self, do_print=True) -> str:
        # Translate the state into colors
        def c(x,y, z, axis):
            """
            Returns the color of the sticker at the given location which is showing on the given axis
            """
            color_index = self.color_index_of_sticker_at(x,y, z, axis)
            colors = '⬜🟨🟦🟩🟥🟧'
            return colors[color_index]

        # Assume the standard orientation (Front face at +x, Up face at +z, Right face at +y)
        l = []
        l.append("      {}{}".format(c(-1,-1,+1,2), c(-1,+1,+1,2)))  # Up face
        l.append("      {}{}".format(c(+1,-1,+1,2), c(+1,+1,+1,2)))
        l.append("{}{} {}{} {}{} {}{}".format(
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
        l.append("{}{} {}{} {}{} {}{}".format(
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
        l.append("      {}{}".format(c(+1,-1,-1,2), c(+1,+1,-1,2)))  # Down face
        l.append("      {}{}".format(c(-1,-1,-1,2), c(-1,+1,-1,2)))
        to_print = "\n".join(l)
        to_print = to_print.replace("  ", "   ")
        if do_print:
            print(to_print)
    
    
# %%

# %%

def __dry_test_cube(cubeclass: type) -> None:
    print(f"Dry running {cubeclass}")
    cube = cubeclass()
    cube.show()
    # for rotation in "UDLRFB":
    #     print(f"{rotation=}")
    #     cube.apply_rotation(rotation)
    #     cube.show()

    data, states = cubeclass.generate_random_data(10, seed=0)
    print(f"{cubeclass.tokenize(data)=}")
    ds = cubeclass.dataset(11, seed=0)
    print(f"{ds[0]=}")

if __name__ == "__main__":
    # __dry_test_cube(CubePuzzle222)
    # __dry_test_cube(CubePuzzle111)
    __dry_test_cube(CubieRepresentation)
    # cube = CubieRepresentation()
    # dl = cube.dataloader(31, batch_size=32, seed=0)
    # for i, (data, state) in enumerate(dl):
    #     print(f"{i=} {data=}")
    #     break
    
# %%
