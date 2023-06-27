"""
Experiment training a model on the Rubiks cube dataset
"""
# %%
import random

from importlib import reload
from typing import List, Optional
from datetime import datetime

import circuitsvis
import numpy as np
import plotly.express as px

import torch as t
import tqdm

import transformer_lens

import wandb
from automata.fa.dfa import DFA
import data_generation
from data_generation import CubePuzzle111
from einops import rearrange, reduce, repeat
from IPython.display import display

from sklearn.linear_model import LogisticRegression
import transformer_lens
from transformer_lens import HookedTransformer

# %%


def refresh():
    reload(data_generation)

refresh()


# %%

device = "cuda"
# %%
transformer_lens.HookedTransformerConfig

# reference_model = HookedTransformer.from_pretrained("gelu-4l", fold_ln=False, center_unembed=False, center_writing_weights=False)
# %%

cfg = transformer_lens.HookedTransformerConfig(
    **{
        "act_fn": "gelu",
        "attention_dir": "causal",
        "attn_only": False,
        "attn_types": None,
        "checkpoint_index": None,
        "checkpoint_label_type": None,
        "checkpoint_value": None,
        "d_head": 32,
        "d_mlp": 768,
        "d_model": 192,
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
        "n_ctx": 32,
        "n_devices": 1,
        "n_heads": 4,
        "n_layers": 4,
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

# %%
cfg.n_params
# %%

model = HookedTransformer(cfg)

# %%

# %%


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


# %%
# aa_train = make_aa_generator(seed=123)
# aa_test = make_aa_generator(seed=456)

model = HookedTransformer(cfg)


def percentage_accepted(prompt=None, model=model):
    return 1.0
    if prompt is None:
        prompt = t.zeros((1000, 1), dtype=t.int64) + ord("B")
    print(f"{prompt.shape=}")
    max_tokens = model.cfg.n_ctx - prompt.shape[1]
    generated_tokens = model.generate(
        input=prompt,
        eos_token_id=0,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=1,
    )

    generated_strings = [
        gen.detokenize(line).strip("\x00") for line in generated_tokens
    ]
    print(generated_strings[:10])
    accepted = [gen.dfa.accepts_input(line) for line in generated_strings]
    gen.pprint_dfa_trajectory(generated_strings[0])
    return sum(accepted) / len(accepted)


# %%
def train_basic_model(
    model, batch_size=64, num_epochs=10_000, seed=123
):
    project_name = f"rubiks-ABC-wandb-develop"
    with wandb.init(project=project_name, job_type="train") as run:
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
        data_loader = CubePuzzle111.dataloader(
            data_length=cfg.n_ctx, batch_size=batch_size, seed=seed
        )

        n_parameters = sum(p.numel() for p in model.parameters())
        parameter_size = (
            n_parameters * model.parameters().__next__().element_size() / 1e6
        )
        print(f"Model has {n_parameters} parameters = {parameter_size} MB")

        """## Model Training"""

        losses = []
        for epoch, (tokens, states) in tqdm.auto.tqdm(
            enumerate(data_loader), total=num_epochs
        ):
            # print(tokens.device, states.device)
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
                accept_frac = percentage_accepted(model=model)
                log_dict["accept_frac"] = accept_frac
                print(f"Epoch {epoch}: {loss.item()} {accept_frac=}")
            run.log(log_dict)
            if epoch > num_epochs:
                break
        return losses

# %%
train = True

timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
model_file_name = f"model_weights_parity_{cfg.n_layers}l_{cfg.d_model}_{timestamp}.pt"

if train:
    losses = train_basic_model(
        model, batch_size=32, num_epochs=20_000, seed=123
    )
    fig = px.line(losses, labels={"x": "Epoch", "y": "Loss"})
    fig.show()
    t.save(model.state_dict(), model_file_name)
else:
    model_file_name = "model_weights_parity_8l_192_2023_06_26-07_40_02_PM.pt"
    model.load_state_dict(t.load(model_file_name))
# %%

dl = CubePuzzle111.dataloader(cfg.n_ctx, batch_size=2)

for tokens, states in dl:
    print(tokens.shape)
    logits = model(tokens.cuda())
    break
# %%
