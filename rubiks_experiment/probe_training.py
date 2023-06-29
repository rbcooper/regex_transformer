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
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import rubiks_generator
from rubiks_generator import CubePuzzle111, CubieRepresentation

device = "cuda"
cfg = tl.HookedTransformerConfig(**base_transformer_config)
model = HookedTransformer(cfg).to(device)
# %% Load data

model_file_name = "checkpoint_rubiks_6l_256d_126c_50000_2023_06_29-09_23_38_PM.pt"
model.load_state_dict(t.load(model_file_name))

# %%

def states_to_one_hot_locations(states: List[List[rubiks_generator.CubieRepresentation]], rotations=True) -> t.Tensor:
    """
    states_len = (seq_len - 1) / 5
    Input: a (batch, states_len) nested list of states
    Output: a (batch, states_len, 8, 8) tensor of one-hot locations
    """
    batch_size = len(states)
    states_len = len(states[0])
    locations = t.zeros(batch_size, states_len, 4, 6, device=device)
    for batch_i, example in enumerate(states):
        for state_i, state in enumerate(example):
            # data = state.inverse_positions_rotations_to_int() if rotations else state.inverse_positions_to_int() # (8, 24)
            data = state.observations()
            for cubie_id in range(locations.shape[2]):
                cubie_location = data[cubie_id]
                locations[batch_i, state_i, cubie_id, rubiks_generator.cube_colors.index(cubie_location)] = 1
                # if state_i == 0:
                #     print(cubie_id, cubie_location)
        # print()
    return locations

# Initialize DataLoader
dl = CubieRepresentation.dataloader(cfg.n_ctx, batch_size=32, seed='probe2', num_workers=0)

for tokens, states in dl:
    tokens = tokens.cuda()
    states = states_to_one_hot_locations(states).cuda()
    print(tokens.shape, states.shape)
    break


# %%

t.cuda.empty_cache()

# Initialize probe: layers (d_model -> cubie_positions cubie_positions)
lr = 1e-3
wd = 0.01
layer = 4
# layers = model.cfg.n_layers
cubie_positions = 8
probe_name = "trivial_probe"
linear_probe = t.randn(
    4,6, model.cfg.d_model, requires_grad=False, device="cuda"
)/np.sqrt(model.cfg.d_model)
linear_probe.requires_grad = True
optimiser = t.optim.AdamW([linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd)
# 0 = when model receives the move
offset = 0

probe_loss_fn = t.nn.CrossEntropyLoss(reduction="none")

n_batches = 1000
# wandb.init(project="othello", name="linear-probe")

# %%

# Make sure our sticker colors returned by state.sticker_colors_to_int() match tokens
for i, (tokens, states) in tqdm.auto.tqdm(enumerate(dl), total=n_batches):
    if i >= n_batches: break

    for i, state in enumerate(states[0]):
        sticker_colors_to_int_values = [rubiks_generator.cube_colors[c] for c in state.sticker_colors_to_int()[0:4]]
        print(f"Color at position 0-3: {sticker_colors_to_int_values}")
        print(f"Tokens: {rubiks_generator.tokenizer.decode(tokens[0, 5*i+1:5*i+6].tolist())}")
        state.show()

    break
        
# %%

# %%

# Initialize DataLoader
dl = CubieRepresentation.dataloader(cfg.n_ctx, batch_size=32, seed='probe')
for i, (tokens, states) in tqdm.auto.tqdm(enumerate(dl), total=n_batches):
    if i >= n_batches: break

    with t.inference_mode():
        _, cache = model.run_with_cache(tokens.cuda(), return_type=None)
        resid_post = cache["resid_post", layer] # (batch, pos, d_model)
    
    # Only take positions where the model receives a move
    resid_post = resid_post[:, offset + 1::5]
    probe_out = einsum(
        resid_post,
        linear_probe,
        "batch pos d_model, cubie_id cubie_loc d_model -> batch pos cubie_id cubie_loc",
    )
    # print(probe_out.shape)

    # predicted probability of each cubie being in each location
    probe_log_probs = probe_out.log_softmax(-1) # (batch, pos, cubie_id, cubie_loc)
    states_one_hot = states_to_one_hot_locations(states).cuda() # (batch, pos, cubie_id, cubie_loc)
    first_sequence = states[0]
    # if i==0:
    #     for state in first_sequence:
    #         state.show()
    # 1 * log prob for the correct-position logits, 0 otherwise. Then reduce over batch
    probe_correct_log_probs = probe_log_probs * states_one_hot # (batch, pos, cubie_id, cubie_loc)
    average_correct_log_probs = reduce(
        probe_correct_log_probs,
        "batch pos cubie_id cubie_loc -> cubie_id",
        "mean"
    ) * cubie_positions # Take the *mean* over batch, pos and *sum* over cubie locations
    loss = -average_correct_log_probs.sum() # sum over cubie_id
    
    loss.backward() # it's important to do a single backward pass for mysterious PyTorch reasons, so we add up the losses - it's per mode and per square.

    optimiser.step()
    optimiser.zero_grad()

    if i % 20 == 0:
        cubie_locations_integer = states_one_hot.argmax(-1)
        accuracy = list(probe_log_probs.argmax(-1).eq(cubie_locations_integer).float().mean((0, 1)).cpu().numpy())
        # wandb.log({
        #     "loss": loss.item(),
        #     "accuracy": accuracy,
        # })
        print(f"loss={loss.item():.3f}, {accuracy=}")
# wandb.finish()
t.save(linear_probe, f"{probe_name}.pth")
# %%

# test linear probe
n_batches = 10
accuracy = []
loss_components = np.zeros((25, 4, 6))
for i, (tokens, states) in tqdm.auto.tqdm(enumerate(dl), total=n_batches):
    if i >= n_batches: break
    with t.inference_mode():
        _, cache = model.run_with_cache(tokens.cuda(), return_type=None)
        resid_post = cache["resid_post", layer]

        resid_post = resid_post[:, 1::5]
        probe_out = einsum(
            resid_post,
            linear_probe,
            "batch pos d_model, cubie_id cubie_loc d_model -> batch pos cubie_id cubie_loc",
        )

        probe_log_probs = probe_out.log_softmax(-1) # (batch, pos, cubie_id, cubie_loc)
        states_one_hot = states_to_one_hot_locations(states).cuda() # (batch, pos, cubie_id, cubie_loc)
        # 1 * log prob for the correct-position logits, 0 otherwise. Then reduce over batch
        probe_correct_log_probs = probe_log_probs * states_one_hot # (batch, pos, cubie_id, cubie_loc)

        loss_components -= reduce(
            probe_correct_log_probs,
            "batch pos cubie_id cubie_loc -> pos cubie_id cubie_loc",
            "mean"
        ).detach().cpu().numpy()
# %%
# Get marginals of loss_components, and create a line graph by pos
loss_components_by_position = reduce(loss_components, "pos cubie_id cubie_loc -> pos cubie_id", "mean")

px.line(
    data_frame=pd.DataFrame(loss_components_by_position),
    # x=np.arange(loss_components_by_position.shape[0]),
    title="Loss components by position",
)

# %%

# Get average loss by cubie_id
loss_components_by_cubie_id = reduce(loss_components, "pos cubie_id cubie_loc -> cubie_id", "mean")
# Make a bar graph
px.bar(
    x=np.arange(loss_components_by_cubie_id.shape[0]),
    y=loss_components_by_cubie_id,
    title="Loss components by cubie_id",
)
# %%
# Now create a heatmap by cubie_id and cubie_loc
loss_components_by_cubie_id_and_cubie_loc = reduce(loss_components, "pos cubie_id cubie_loc -> cubie_id cubie_loc", "mean")

px.imshow(
    loss_components_by_cubie_id_and_cubie_loc,
    title="Loss components by cubie_id and cubie_loc",
    labels=dict(x="cubie_loc", y="cubie_id"),
)
# %%
# print(probe_log_probs.shape) # (batch, pos, cubie_id, cubie_loc)

# probe_predictions = probe_log_probs.argmax(-1)[:, 0, 1] # (batch, pos 0, cubie_id 0)
# # Get frequency of predictions at position 0
# px.histogram(probe_predictions.cpu().numpy(), title="Predictions at position 0")
# %%
