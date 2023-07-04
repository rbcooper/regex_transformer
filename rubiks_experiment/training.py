"""
Experiment training a model on the Rubiks cube dataset
"""
# %%
import random
from datetime import datetime

from importlib import reload
from typing import List, Optional, Tuple

import circuitsvis
import numpy as np
import plotly.express as px
import rubiks_generator

import torch as t
import tqdm

import transformer_lens

import wandb
from automata.fa.dfa import DFA

# from rubiks_generator import CubePuzzle111, CubieRepresentation
from einops import rearrange, reduce, repeat
from frozendict import frozendict
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer


# %% refresh(), device, etc.


def refresh():
    reload(rubiks_generator)


refresh()


device = "cuda"
# %% unused


# reference_model = HookedTransformer.from_pretrained("gelu-4l", fold_ln=False, center_unembed=False, center_writing_weights=False)
# %% Make model

base_transformer_config = frozendict(
    {
        "act_fn": "gelu",
        "attention_dir": "causal",
        "attn_only": False,
        "attn_types": None,
        "checkpoint_index": None,
        "checkpoint_label_type": None,
        "checkpoint_value": None,
        "d_head": 48,
        "d_mlp": 1024,
        "d_model": 256,
        "d_vocab": 256,
        "d_vocab_out": 256,
        "device": "cuda",
        "eps": 1e-05,
        "final_rms": False,
        "from_checkpoint": False,
        "gated_mlp": False,
        "init_mode": "gpt2",
        "init_weights": True,
        # 'initializer_range': 0.035355339059327376,
        "model_name": "regex-tester",
        "n_ctx": 126,
        "n_devices": 1,
        "n_heads": 6,
        "n_layers": 6,
        # 'n_params': 12582912,
        "normalization_type": "LN",
        "original_architecture": "neel",
        "parallel_attn_mlp": False,
        "positional_embedding_type": "standard",
        "rotary_dim": None,
        "scale_attn_by_inverse_layer_idx": False,
        "seed": None,
        #  'tokenizer_name': 'NeelNanda/gpt-neox-tokenizer-digits',
        "use_attn_result": False,
        "use_attn_scale": True,
        "use_hook_tokens": False,
        "use_local_attn": False,
        "use_split_qkv_input": False,
        "window_size": None,
    }
)

# model_sweep_config = {
#     "n_layers": {"values": [2, 3, 4, 6, 8]},
#     "d_model": {"values": [64, 128, 192, 256]},
#     "n_heads": {"values": [2, 3, 4, 6]},
#     "d_mlp": {"values": [256, 512, 768]},
# }


cfg = transformer_lens.HookedTransformerConfig(**base_transformer_config)

cfg.n_params


# %% Define loss functions


def loss_fn(logits, tokens, return_per_token=False):
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    # assert False
    logits = logits[:, :-1]  # ignore last logit
    tokens = tokens[:, 1:]  # ignore first token (bos token)
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    loss = -correct_log_probs.mean()  # mean over batch and pos
    return loss


def color_loss(logits, tokens):
    logits = logits.clone()[:, :-1]  # ignore last logit
    tokens = tokens.clone()[:, 1:]  # ignore first token (bos token)
    # ignore every 5th token; it's a direction
    mask = t.tensor([x % 5 != 0 for x in range(tokens.shape[1])], dtype=t.bool)
    logits = logits[:, mask]
    tokens = tokens[:, mask]
    # print(logits.shape, tokens.shape)
    # print(rubiks_generator.tokenizer.decode(list(tokens[0])))
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    loss = -correct_log_probs.mean()  # mean over batch and pos
    return loss


# %%
# aa_train = make_aa_generator(seed=123)
# aa_test = make_aa_generator(seed=456)

model = HookedTransformer(cfg)
model.tokenizer = rubiks_generator.tokenizer


# %% train model function
def train_basic_model(
    model,
    batch_size=64,
    num_epochs=10_000,
    seed=123,
    save_every=None,
    data_generator=rubiks_generator.generate_2x2x2_cube_up_free,
):
    project_name = f"rubiks-world-representation"
    with wandb.init(project=project_name, entity="alighnment", job_type="train") as run:
        print(f"Batch size: {batch_size}")
        lr = 1e-4
        betas = (0.9, 0.95)
        max_grad_norm = 1.0
        wd = 0.01
        optimizer = t.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, weight_decay=wd
        )
        scheduler = t.optim.lr_scheduler.LambdaLR(
            optimizer, lambda i: min(i / 100, 1.0)
        )
        data_loader = rubiks_generator.make_dataloader(
            data_generator,
            batch_size=batch_size,
            seq_length=cfg.n_ctx - 1,
            num_workers=8,
            seed=0,
        )

        n_parameters = sum(p.numel() for p in model.parameters())
        parameter_size = (
            n_parameters * model.parameters().__next__().element_size() / 1e6
        )
        print(f"Training with config: \n{cfg}")
        print(f"Model has {n_parameters} parameters = {parameter_size} MB")

        """## Model Training"""
        model.train()
        losses = []
        for epoch, (tokens, states) in tqdm.auto.tqdm(
            enumerate(data_loader), total=num_epochs
        ):
            # print(tokens)
            tokens = tokens.cuda()
            logits = model(tokens)
            loss = loss_fn(logits, tokens)
            loss.backward()
            if max_grad_norm is not None:
                t.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            losses.append(loss.item())
            log_dict = dict(
                loss=loss.item(),
            )
            if epoch % 100 == 0:
                this_color_loss = color_loss(logits, tokens)
                log_dict["color_loss"] = this_color_loss.item()
                print(f"Epoch {epoch}: {loss.item()} {this_color_loss=}")
            if save_every is not None and epoch % save_every == 0 and epoch > 0:
                timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                model_file_name = f"models/checkpoint_rubiks_{cfg.n_layers}l_{cfg.d_model}d_{cfg.n_ctx}c_{epoch}_{timestamp}.pt"
                t.save(model.state_dict(), model_file_name)
            run.log(log_dict)
            if epoch > num_epochs:
                break
        return losses


# %% train model

experiment_name = "no_us_first_10"


def generate_2x2x2_cube_data_no_beginning_Us(
    shape: int, rng: np.random.Generator
) -> Tuple[list, list]:
    while True:
        (data, state) = rubiks_generator.generate_2x2x2_cube_up_free(
            shape, rng=rng
        )
        if rubiks_generator.tokenizer.token_to_id("U ") not in data[:10]:
            return (data, state)


# %%

train_data_generator = generate_2x2x2_cube_data_no_beginning_Us

if __name__ == "__main__":
    train = False

    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    model_file_name = f"models/model_weights_rubiks_{cfg.n_layers}l_{cfg.d_model}_{timestamp}_{experiment_name}.pt"

    if train:
        losses = train_basic_model(
            model,
            batch_size=32,
            num_epochs=200_000,
            seed=123,
            save_every=10000,
            data_generator=train_data_generator,
        )
        fig = px.line(losses, labels={"x": "Epoch", "y": "Loss"})
        # fig.show()
        t.save(model.state_dict(), model_file_name)
    else:
        model_file_name = "/home/ubuntu/regex_transformer/rubiks_experiment/models/model_weights_rubiks_6l_256_2023_07_03-03_24_52_PM_no_us_first_10.pt"
        model.load_state_dict(t.load(model_file_name))
# %%

test_data_generator = rubiks_generator.generate_2x2x2_cube_up_free
if __name__ == "__main__":
    with t.inference_mode():
        model.eval()
        dl = rubiks_generator.make_dataloader(
            test_data_generator,
            batch_size=32,
            seq_length=cfg.n_ctx - 1,
            num_workers=8,
            seed=123,
        )
        for i, (tokens, states) in enumerate(dl):
            tokens = tokens.cuda()
            print(tokens.shape)
            actual = rubiks_generator.tokenizer.decode(list(tokens[0]))
            print(f"actual:{actual}")
            logits = model(tokens)
            this_color_loss = color_loss(logits, tokens).item()
            print(f"{this_color_loss=:.3f}")
            most_likely_tokens = list(logits.argmax(-1)[0])
            s = rubiks_generator.tokenizer.decode(most_likely_tokens)
            print(f"predicted:  {s}")
            if i > 20:
                break
# %%

# Profile model
# tokens = tokens.clone()
# from torch.profiler import profile, ProfilerActivity, record_function


# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     model(tokens)

# output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
# print(output)
# prof.export_chrome_trace("trace.json")
# %%


# %%

# %%

# # Test with missing various first letters

train_data_generator = generate_2x2x2_cube_data_no_beginning_Us

simple_moves = (m for m in rubiks_generator.all_moves if "2" not in m and "'" not in m)

# Biased data sets
filtered_1_biased_datasets = {}


def train_biased_model(
    moves=simple_moves,
    start_filter_length=1,
    base_generator=rubiks_generator.generate_2x2x2_cube_up_free,
):
    moves_to_filtered_model = {}
    seed = 783248792
    for move in moves:
        move_id = rubiks_generator.tokenizer.token_to_id(move)
        print(f"Training model with first {start_filter_length} {move}s missing")

        def train_data_generator_biased(
            shape: int, rng: np.random.Generator
        ) -> Tuple[list, list]:
            while True:
                (data, state) = base_generator(shape, rng=rng)
                if move_id not in data[: start_filter_length + 1]:
                    return (data, state)

        model = HookedTransformer(cfg)
        model.tokenizer = rubiks_generator.tokenizer
        timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        move_nice = move.replace(" ", "").replace("'", "p")
        model_file_name = f"models/model_weights_rubiks_{cfg.n_layers}l_{cfg.d_model}_{timestamp}_filterdfirst_{start_filter_length}_{move_nice}.pt"
        num_epochs = 20_000
        losses = train_basic_model(
            model,
            batch_size=32,
            num_epochs=num_epochs,
            seed=seed,
            save_every=10000,
            data_generator=train_data_generator_biased,
        )
        fig = px.line(losses, labels={"x": "Epoch", "y": "Loss"})
        # fig.show()
        print(f"saving {model_file_name}")
        t.save(model.state_dict(), model_file_name)
        moves_to_filtered_model[move] = model
    model_dict_to_save = {a: m.state_dict() for a, m in moves_to_filtered_model.items()}
    t.save(
        model_dict_to_save,
        f"models/dict_model_weights_rubiks_{cfg.n_layers}l_{cfg.d_model}_{timestamp}_filterdfirst_{start_filter_length}_all_epoch_{num_epochs}.pt",
    )
    return moves_to_filtered_model


if __name__ == "__main__":
    train = False
    if train:
        move_to_trained_filtered = train_biased_model(start_filter_length=1)
    else:
        "models/dict_model_weights_rubiks_6l_256_2023_07_04-12_02_27_PM_filterdfirst_1_all_epoch_20000.pt"
# %%
