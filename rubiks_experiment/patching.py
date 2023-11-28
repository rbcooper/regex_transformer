# %%
from training import base_transformer_config
import numpy as np
import torch as t
import plotly.express as px
from einops import rearrange, reduce, repeat, einsum
from sklearn.linear_model import LogisticRegression
import transformer_lens as tl
from transformer_lens import HookedTransformer
import tqdm
from typing import List
import wandb
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import rubiks_generator
from rubiks_generator import CubePuzzle111, CubieRepresentation, tokenizer, CubeMove

device = "cuda"
cfg = tl.HookedTransformerConfig(**base_transformer_config)
model = HookedTransformer(cfg).to(device)
# %% Load data

model_file_name = (
    "models/checkpoint_rubiks_6l_256d_126c_110000_2023_07_04-07_25_09_PM.pt"
)
model.load_state_dict(t.load(model_file_name))

# %%
t.cuda.empty_cache()

cube_colors = "⬜🟨🟩🟦🟥🟧"


def run_model(move_seq, model=model, prepend_eos=True):
    if prepend_eos:
        move_seq = "<eos>" + move_seq
    with t.no_grad():
        ids = tokenizer.encode(move_seq)
        if 0 in ids[prepend_eos:]:
            print("Warning: 0 in model output")
        ret = model(t.tensor(ids))
        return ret


# %%
def top_k(model_output, k=1):
    # print(model_output.shape)
    model_output = model_output.squeeze(0)
    values, indices = t.topk(model_output[-1, :], k=k, dim=-1)
    values = values.detach().cpu().numpy()
    indices = tokenizer.decode(list(indices.detach().cpu().numpy())).split()
    # print(topk)
    return dict(zip(indices, values))


# %%

# %%
cube = CubieRepresentation()
cube.show()
rotation = "U "
print(cube.after_move(rotation).observations())
top_k(run_model(rotation + "⬜⬜🟩"), k=3)
# %%


tokens, states = rubiks_generator.generate_222_cube_data_raw_face(
    125, np.random.default_rng(0)
)
# %%
# tokens, states
# %%
tokenizer.decode(list(tokens[: 76 + 1]))
# %%

# Hypothesis: the model’s predictions are based on memorizing move sequences that take up-face stickers back to the up-face, attending to the shortest suffix that meets this criterion, then copying the previous color to the output.
# If the hypothesis is correct, we should see tokens after U attending to their counterparts right before the U,
# then copying the values.
token_prefix = tokens[: 76 + 1]
print(tokenizer.decode(list(token_prefix)))
edited_token_prefix = token_prefix.clone()
edited_token_prefix[74] = tokenizer.token_to_id("⬜")

print("Unedited:")
top_k(model(token_prefix), k=3)
# %%
print(tokenizer.decode(list(edited_token_prefix)))
print("Edited:")
top_k(model(edited_token_prefix), k=3)

# We see the ⬜ token now dominates, which supports our hypothesis.

# %%
model_diff = model(token_prefix) - model(edited_token_prefix)
model_diff[0, -1, :].std()
# %%

# Next test: generate a sequence ending in L'... L. Model should copy the color from 10 moves ago to positions L+1 and L+3.
rng = np.random.default_rng(0)
while True:
    tokens, states = rubiks_generator.generate_222_cube_data_raw_face(125, rng)
    L_token = tokenizer.token_to_id("L ")
    Lp_token = tokenizer.token_to_id("L'")
    if tokens[71] == L_token and tokens[76] == Lp_token:
        break

print(tokenizer.decode(list(tokens[: 76 + 1])))
# Now the model should copy the 🟨.
# %%
token_prefix = tokens[: 76 + 1]
edited_token_prefix = token_prefix.clone()
edited_token_prefix[67] = tokenizer.token_to_id("🟦")

print("Unedited:")
top_k(model(token_prefix), k=3)
# %%
print("Edited:")
top_k(model(edited_token_prefix), k=3)
# So the model does copy the color in position 67 to position 77.
# %%

# Let's try a longer sequence, where the model doesn't see the piece for 4 moves / ~20 tokens.
rng = np.random.default_rng(0)
while True:
    tokens, states = rubiks_generator.generate_222_cube_data_raw_face(125, rng)
    L_token = tokenizer.token_to_id("L ")
    Lp_token = tokenizer.token_to_id("L'")

    if (
        tokens[56] == Lp_token
        and tokens[76] == L_token
        and all(t in tokenizer.encode("U F R U'B'R'").ids for t in tokens[61:76:5])
    ):
        break

print(tokenizer.decode(list(tokens[: 76 + 1])))
# Now the token it needs to attend to is 🟧 at position 52.
# %%

token_prefix = tokens[: 76 + 1]
edited_token_prefix = token_prefix.clone()
edited_token_prefix[52] = tokenizer.token_to_id("🟦")

print("Unedited:")
top_k(model(token_prefix), k=3)
# %%
print("Edited:")
top_k(model(edited_token_prefix), k=3)

# It still seems to be copying the color from 52 to 77.
# %%


# Function that accepts sequences that have a sticker visible, invisible from move 10 to move 19 inclusive, then visible again.
def get_reappearance_allowed_rotations_fn(leaving_move=10, reappearing_move=20):
    rot_to_leaving_sticker_dict = {
        "F ": (1, 1),
        "F'": (1, -1),
        "R ": (-1, 1),
        "R'": (1, 1),
        "B ": (-1, -1),
        "B'": (-1, 1),
        "L ": (1, -1),
        "L'": (-1, -1),
    }

    def reappearance_allowed_rotations_fn(
        states: List[CubieRepresentation],
        rotations: List[CubeMove],
        this_rotation=CubeMove,
    ) -> bool:
        """
        when the proposed move is n (0-indexed), we have n+1 post-move states, n-1 past rotations, and this rotation.
        After current rotation, we go to state n
        """
        n = len(states) - 1
        assert leaving_move >= 1
        if n < leaving_move or n > reappearing_move:
            return True
        elif n == leaving_move:
            # cube was visible and is now invisible
            # Therefore this move must be an F/F', R/R', B/B', or L/L' move
            ret = this_rotation in ["F ", "F'", "R ", "R'", "B ", "B'", "L ", "L'"]
            print(
                f"Sticker {rot_to_leaving_sticker_dict[this_rotation]} is leaving at move {n} ({this_rotation})"
            )
            x, y = rot_to_leaving_sticker_dict[this_rotation]
            cubie_id = states[leaving_move - 1].get_cubie_id_of_piece_at(x, y, 1)
            # states[leaving_move - 1].show()
            sticker_id = (
                cubie_id * 3 + states[leaving_move - 1].cubie_rotations[cubie_id][2]
            )  # z axis after move 9
            print(
                f"Sticker ID: {sticker_id} on cubie {cubie_id}, color {rubiks_generator.cube_colors[states[leaving_move - 2].sticker_colors[sticker_id]]}"
            )
            print(
                f"Sticker coming from {states[leaving_move - 1].get_position_of_sticker_id(sticker_id)}"
            )
            print(
                f"Sticker going to {states[leaving_move].get_position_of_sticker_id(sticker_id)}"
            )
            # if ret:
            # print(f"Sticker going to {states[-1]}")
            return ret
        elif leaving_move < n <= reappearing_move:
            # Get the ID of the sticker that was taken off the top face at move 10
            x, y = rot_to_leaving_sticker_dict[rotations[leaving_move]]
            # print(f"Sticker leaving from {x},{y},1 with move {rotations[leaving_move - 1]}")
            cubie_id = states[leaving_move - 1].get_cubie_id_of_piece_at(x, y, 1)
            # states[leaving_move - 1].show()
            sticker_id = (
                cubie_id * 3 + states[leaving_move - 1].cubie_rotations[cubie_id][2]
            )  # z axis after move 9

            x, y, z, axis = states[-1].get_position_of_sticker_id(sticker_id)
            if n == reappearing_move - 1:
                # Ensure the sticker is not on the bottom or top
                ret = axis != 2
            elif n == reappearing_move:
                # Ensure the sticker is on the up face
                ret = z == 1 and axis == 2
                if ret:
                    print(
                        f"Sticker returning with move {this_rotation} to {x},{y},{z} with axis {axis}"
                    )
            else:
                # Ensure the sticker is not on the up face
                ret = not (z == 1 and axis == 2)
            if ret:
                print(
                    f"At move {n} after rotation {this_rotation}, sticker {sticker_id} is at {x},{y},{z} with axis {axis}"
                )
            return ret

    return reappearance_allowed_rotations_fn


# %%
rng = np.random.default_rng(0)
reappearance_data = rubiks_generator.generate_222_cube_data_raw_face(
    125, rng, allowed_rotations_fn=get_reappearance_allowed_rotations_fn(1, 3)
)
# %%
print(tokenizer.decode(list(reappearance_data[0])))

for state in reappearance_data[1]:
    state.show()
    print()
# %%
