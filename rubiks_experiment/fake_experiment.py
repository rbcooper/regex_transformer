# %%
import transformer_lens as tl
from transformer_lens import HookedTransformer
import numpy as np
import torch as t
import plotly.express as px
from einops import rearrange, reduce, repeat, einsum


# %%

# This is a fake dataset where the cube is always in the solved position
# The color queries are random
always_solved_dataset = load_dataset("rubiks_corners_always_solved", split="test", total_tokens=10)
assert always_solved_dataset[0]["text"] == "++-y " # TODO

# A dataset with random moves only
no_queries_dataset = load_dataset("rubiks_corners_no_queries", split="test")

# Dataset where the the first part are moves, then the rest are state query/answers

moves_then_queries_dataset = rubiks_datasets.moves_then_queries(moves=5, total_tokens=126)



# %%

# Training:

# rubiks_models

@magic_model
def color_query_probe_training():
    model_cfg = base_transformer_config,
    dataset = dataset, # has metadata in it?

    model_cfg = base_transformer_config


train_model(dataset=dataset,
            )

# %%

# def is_color_token_mask(Integer[torch.Tensor, "batch pos"]) -> Boolean[torch.Tensor, "batch pos"]

# def is_query_token_mask(Integer[torch.Tensor, "batch pos"]) -> Boolean[torch.Tensor, "batch pos"]

# def is_invalid_token_mask(Integer[torch.Tensor, "batch pos"]) -> Boolean[torch.Tensor, "batch pos"]

# def is_invalid_sequence_mask(Integer[torch.Tensor, "batch pos"]) -> Boolean[torch.Tensor, "batch"]

# def is_valid_query_color_sequence(tokens: Union[str, Integer[torch.Tensor, "pos"]]) -> bool:

# def highlight_invalid(tokens: Union[str, Integer[torch.Tensor, "pos"]]) -> str:

# def count_invalid(tokens: Union[str, Integer[torch.Tensor, "pos"]]) -> int:

# def ensure_tokenized(tokens: Union[str, Integer[torch.Tensor, "pos"]]) -> Integer[torch.Tensor, "pos"]:


dataset_config = {
    "name": "n_moves_then_query_dataset",
    "kwargs" = {
        n_moves: 5,
    }
}



# %%

def test_model_handles_solved_cube(model):
    [ for s in model(always_solved_dataset[:30]["tokens"]) if not is_valid_sequence(s)]




# %%

cfg = tl.HookedTransformerConfig(**base_transformer_config)
model = HookedTransformer(cfg).to(device)


model_file_name = (
    "models/checkpoint_rubiks_6l_256d_126c_180000_2023_07_23-12_22_16_PM.pt"
)
model.load_state_dict(t.load(model_file_name))

model.run_with_cache()

# consider: get all memory at once

def train_scikit_probe(model: HookedTransformer, dataset: Dataset, scikit_learn_model):
    pass


def extract_feature(model: HookedTransformer, scikit_learn_model):


    pass





# probe.partial_fit(X, y[, classes, sample_weight])
# X : {array-like, sparse matrix}, shape = [n_samples, n_features]
#   Training vectors, where n_samples is the number of samples and n_features is the number of features.
# y : array-like, shape = [n_samples]
#   Target values.
# classes : array-like, shape = [n_classes]
#   List of all the classes that can possibly appear in the y vector.
#   Must be provided at the first call to partial_fit, can be omitted in subsequent calls.





# %%


# Tests that a





# State is inferred from the tokens
dataloader(test)



# %%

