"""
Experiment training a model on the Rubiks cube dataset
"""
# %%
import random

from importlib import reload
from typing import List, Optional
from datetime import datetime
from frozendict import frozendict

import circuitsvis
import numpy as np
import plotly.express as px

import torch as t
import tqdm

import transformer_lens

import wandb
from automata.fa.dfa import DFA
import rubiks_generator
# from rubiks_generator import CubePuzzle111, CubieRepresentation
from einops import rearrange, reduce, repeat
from IPython.display import display

from sklearn.linear_model import LogisticRegression
import transformer_lens
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



# %% train model function
def train_basic_model(
    model, batch_size=64, num_epochs=10_000, seed=123, save_every=None
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
        data_loader = rubiks_generator.CubieRepresentation.dataloader(
            data_length=cfg.n_ctx, batch_size=batch_size, seed=seed
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
        for epoch, (tokens, states) in tqdm.auto.tqdm(enumerate(data_loader), total=num_epochs):
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
            if save_every is not None and epoch % save_every == 0:
                timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                model_file_name = f"models/checkpoint_rubiks_{cfg.n_layers}l_{cfg.d_model}d_{cfg.n_ctx}c_{epoch}_{timestamp}.pt"
                t.save(model.state_dict(), model_file_name)
            run.log(log_dict)
            if epoch > num_epochs:
                break
        return losses

# %% train model 

if __name__ == "__main__":
    train = True

    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    model_file_name = f"models/model_weights_rubiks_{cfg.n_layers}l_{cfg.d_model}_{timestamp}.pt"

    if train:
        losses = train_basic_model(
            model, batch_size=32, num_epochs=200_000, seed=123, save_every=10000
        )
        fig = px.line(losses, labels={"x": "Epoch", "y": "Loss"})
        fig.show()
        t.save(model.state_dict(), model_file_name)
    else:
        model_file_name = "/home/ubuntu/regex_transformer/rubiks_experiment/models/checkpoint_rubiks_6l_256d_126c_300000_2023_06_27-11_08_17_PM.pt"
        model.load_state_dict(t.load(model_file_name))
# %%

if __name__ == "__main__":
    with t.inference_mode():
        model.eval()
        dl = rubiks_generator.CubieRepresentation.dataloader(cfg.n_ctx, batch_size=32)

        for tokens, states in dl:
            tokens = tokens.cuda()
            print(tokens.shape)
            print(rubiks_generator.tokenizer.decode(list(tokens[0])))
            logits = model(tokens)
            this_color_loss = color_loss(logits, tokens).item()
            print(f"{this_color_loss=:.3f}")
            most_likely_tokens = list(logits.argmax(-1)[0])
            s = rubiks_generator.tokenizer.decode(most_likely_tokens)
            print(s)
            break
# %%