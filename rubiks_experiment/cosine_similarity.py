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
from rubiks_generator import CubePuzzle111, CubieRepresentation

device = "cuda"
cfg = tl.HookedTransformerConfig(**base_transformer_config)
model = HookedTransformer(cfg).to(device)

# %% Load data

model_file_name = "models/checkpoint_rubiks_6l_256d_126c_50000_2023_06_29-09_23_38_PM.pt"
model.load_state_dict(t.load(model_file_name))

# %%
t.cuda.empty_cache()

n_batches = 1

# Initialize DataLoader and get data
dl = CubieRepresentation.dataloader(cfg.n_ctx, batch_size=32, seed='probe', num_workers=0)
for i, (tokens, states) in tqdm.auto.tqdm(enumerate(dl), total=n_batches):
    if i >= n_batches: break
    tokens = tokens.cuda()

with t.inference_mode():
    logits, cache_random = model.run_with_cache(tokens)


def average_cosine_similarity(tokens, hook, layer, pos=56):
    logits, cache = model.run_with_cache(tokens)
    activations = cache[hook, layer]
    sv1 = activations[:, pos:pos+1, :]
    sv2 = rearrange(sv1, 'b 1 d_model -> 1 b d_model')
    all_cos_sims = t.cosine_similarity(sv1, sv2, dim=-1)
    # Set diagonal values to nan, to correct for the fact that
    # the cosine similarity of a vector with itself is 1
    all_cos_sims[t.arange(all_cos_sims.shape[0]), t.arange(all_cos_sims.shape[0])] = t.nan
    return all_cos_sims.nanmean().cpu().numpy()

print("Average cos sim of different seqs: ", average_cosine_similarity(tokens, 'resid_post', 3))

new_tokens = repeat(tokens[0], 'seq -> batch seq', batch=12).clone()
# Take sample 0, then edit tokens 51 and 56 to be X and X' for various faces X
rng = np.random.default_rng()
for i in range(new_tokens.shape[0]):
    face = "UDLRFB"[i // 2]
    token_51 = rubiks_generator.tokenizer.encode(f"{face} " if i % 2 == 0 else f"{face}'")[0]
    token_56 = rubiks_generator.tokenizer.encode(f"{face}'" if i % 2 == 0 else f"{face} ")[0]
    new_tokens[i, 51] = token_51
    new_tokens[i, 56] = token_56

print("Average cos sim of isomorphisms after same seqs", average_cosine_similarity(new_tokens, 'resid_post', 3))
# Cosine similiarity is not significantly higher for isomorphisms than for random states,
# even when the states are the same except for the last two tokens
# %%
