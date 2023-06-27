# %%
from torch.utils.data import Dataset, DataLoader
import tokenize
import tokenizers
import numpy as np
import torch as t

from typing import Tuple, Optional

# %%
class CubePuzzle():
    def __init__():
        pass

    def sticker_value(self, face, row, col):
        pass

    def cubie_position(self, cubie_id:int):
        pass

    def apply_rotation(self, face, direction):
        pass

    @staticmethod
    def generate_random_data(self, data_length:int) -> Tuple[list, list]:
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
    sticker_values:tuple

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
    
    def apply_rotation(self, rotation_name:str):
        """
        For the 1x1x1 cube, there are only 6 rotations.
        """
        # When applying a rotation, the NEW ith sticker value is the OLD rotation[i]'th sticker value.
        self.sticker_values = tuple(self.sticker_values[i] for i in self.rotations[rotation_name])

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

    @staticmethod
    def dataset(data_length: int, seed=None) -> Dataset:
        """A torch dataset yielding (random_word, dfa_state) pairs)
        """
        return CubePuzzleDataset(data_length, seed=seed)

    @staticmethod
    def dataloader(data_length: int, batch_size: int, seed=None) -> DataLoader:
        return DataLoader(CubePuzzle111.dataset(data_length, seed=seed), batch_size=batch_size)

    @classmethod
    def generate_random_data(self, data_length:int, seed:int) -> Tuple[list, list]:
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
    def tokenize(self, data:list, prepend_eos=True):
        """
        Returns a tensor with interlaced observations and rotations.
        Observations are ASCII codes for numbers 0-5, rotations are letters.
        Also prepends the EOS token 0.
        """
        for i in range(len(data)):
            if data[i] in self.rotation_names:
                data[i] = ord(data[i])
            else:
                data[i] = ord(str(data[i]))
        if prepend_eos:
            data = [0] + data
        return t.tensor(data, dtype=t.int64)


# %%
class CubePuzzleDataset(Dataset):
    def __init__(
        self, data_length: int, *, seed:Optional[int] = None, prepend_eos:bool = True
    ):
        self.data_length = data_length
        self.seed = seed
        self.prepend_eos = prepend_eos

    def __len__(self):
        return (2**31) - 1

    def __getitem__(self, idx):
        data, states = CubePuzzle111.generate_random_data(self.data_length - self.prepend_eos, seed=idx)
        data = CubePuzzle111.tokenize(data, prepend_eos=self.prepend_eos)
        return data , states


# %%
if __name__ == "__main__":
    cube = CubePuzzle111()
    cube.show()
    for rotation in "UDLRFB":
        print(rotation)
        cube.apply_rotation(rotation)
        cube.show()

    data, states = CubePuzzle111.generate_random_data(11, seed=0)
    print(CubePuzzle111.tokenize(data))
    ds = CubePuzzle111.dataset(10, seed=0)
    print(ds[0])
# %%
