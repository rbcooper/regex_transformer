# %%
import tokenize
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as t

from tokenizers import AddedToken, models, Tokenizer
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy

from einops import rearrange, repeat


cube_colors = "⬜🟨🟩🟦🟥🟧"
basic_moves = "UDLRFB"

all_moves = []
for m in basic_moves:
    all_moves.append(f"{m} ")
    all_moves.append(f"{m}'")
    all_moves.append(f"{m}2")

vocab = [*cube_colors, *all_moves]

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
    def dataloader(cls, data_length: int, batch_size: int, seed=None, num_workers=4) -> DataLoader:
        def my_collate(batch):
            data = t.stack([item[0] for item in batch])
            cube_state = [item[1] for item in batch]
            return data, cube_state

        return DataLoader(
            cls.dataset(data_length, seed=seed),
            batch_size=batch_size,
            collate_fn=my_collate,
            num_workers=num_workers,
        )

    @classmethod
    def generate_random_data(cls, data_length: int, seed: int) -> Tuple[list, list]:
        """
        Generates a sequence of (observation, rotation, observation, rotation, ...),
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
            states.append(cube)
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

    rotation_names = [s + ' ' for s in "U D L R F B".split()]
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
        # print(
        #     f"dataset to generate data of length {self.data_length - self.prepend_eos}"
        # )

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
    def __init__(self):
        # cubie i has stickers 3i to 3i+2
        # positions 1, 3, 5, 7 are on the top face
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

    def apply_rotation(self, rotation: str):
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
                (
                    self.cubie_rotations[i, (axis + 1) % 3],
                    self.cubie_rotations[i, (axis + 2) % 3],
                ) = (
                    self.cubie_rotations[i, (axis + 2) % 3],
                    self.cubie_rotations[i, (axis + 1) % 3],
                )

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
        ret = [None] * 4
        # # Gets sticker values of up face, in order
        # for i in range(len(self.cubie_locations)):
        #     if self.cubie_locations[i][2] == 1:
        #         # this is on the top face; get the z sticker
        #         # 0 for (-1, -1, 1); 1 for (-1, 1, 1) etc.
        #         cubie_location = (self.cubie_locations[i][0] == 1) * 2 + (
        #             self.cubie_locations[i][1] == 1
        #         )
        #         z_sticker = 3 * i + list(self.cubie_rotations[i]).index(2)
        #         ret[cubie_location] = cube_colors[self.sticker_colors[z_sticker]]

        for i in range(len(ret)):
            ret[i] = cube_colors[self.color_index_of_sticker_at(2 * (i // 2) - 1, 2 * (i % 2) - 1, 1, axis=2)]
        # print(f"observations={ret}")
        # for i in range(len(ret)):
        #     print(cube_colors[
        #         self.color_index_of_sticker_at(2 * (i // 2) - 1, 2 * (i % 2) - 1, 1, axis=2)
        #     ])
        # for i in range(len(ret)):
        #     print(f"{i=}")
        #     assert ret[i] is not None
        #     assert ret[i] == cube_colors[
        #         self.color_index_of_sticker_at(2 * (i // 2) - 1, 2 * (i % 2) - 1, 1, axis=2)
        #     ]
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
        # Returns a 24-element list of this cube's sticker colors.
        ret = []
        for x in range(-1,2,2):
            for y in range(-1,2,2):
                for z in range(-1,2,2):
                    for axis in range(3):
                        ret.append(self.color_index_of_sticker_at(x,y,z,axis))
        return np.array(ret)



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
        rotations = []  # rng.choice(cls.rotation_names, size=data_length // 5)
        cube = cls()
        states = []
        observations = []
        for i in range(n_observations):
            axis = rng.integers(3)
            face = rng.choice([-1, 1])
            direction = rng.choice([-1, 1])
            rotation_name = "F B R L U D".split()[axis * 2 + (face == -1)]
            if direction == -1:
                rotation_name += "'"
            else:
                rotation_name += " "
            rotations.append(rotation_name)
            cube.apply_rotation(axis, face, direction)
            # TODO find some way to log states without breaking dataloader
            # states.append(cube.sticker_values)
            # observations is list of strings
            observations.append(cube.observations())
            states.append(deepcopy(cube))
            # print(len(states))

        # interlace observations and rotations
        data = []
        for r, o in zip(rotations, observations):
            # print(f"{type(r)=}, {type(o)=}")
            data.append(r)
            data.extend(o)
        return data, states  # states

    def show(self, do_print=True) -> str:
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
        if do_print:
            print(to_print)


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
    dl = cubeclass.dataloader(126, batch_size=3, seed=0, num_workers=0)
    for i, (data, states) in enumerate(dl):
        for pos in range(25):
            state = states[0][pos]
            datum= data[0][5*pos+1:5*pos+6] # uses observations
            datum = tokenizer.decode(list(datum))
            print(f"{i=} {datum=}")
            print(f"{state.observations()=}")
            state.show() # uses color_index_of_sticker_at
        break


if __name__ == "__main__":
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
