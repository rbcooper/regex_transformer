"""
Experiment training a model on the parity task
"""
# %%
import random

from importlib import reload
from typing import List, Optional

import circuitsvis
import dfa_generator
import numpy as np
import plotly.express as px

import torch as t
import tqdm

import transformer_lens

import wandb
from automata.fa.dfa import DFA
from dfa_generator import DfaGenerator
from einops import rearrange, reduce, repeat
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer

import models

# %%


def refresh():
    reload(dfa_generator)
    global gen
    gen = dfa_generator.DfaGenerator.from_regex("((B|C)*AB*A)*(B|C)*A?B*")


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

# 2 can only appear after an even number of 0s
gen = dfa_generator.DfaGenerator.from_regex("((B|C)*AB*A)*(B|C)*A?B*")
# gen = dfa_generator.DfaGenerator.from_regex('(ROBERT|THOMAS|X)*')
data_loader = gen.batches_and_states_gen(word_len=32, batch_size=32)
batch, states = data_loader.__next__()
batch.shape, states.shape
dfa_generator.display_fa(gen.dfa)
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


def percentage_accepted(prompt=None, model=model, gen=gen):
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
    model, gen: dfa_generator.DfaGenerator, batch_size=64, num_epochs=10_000, seed=123
):
    project_name = f"parity-ABC-wandb-develop"
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
        data_loader = gen.dataloader(
            word_length=cfg.n_ctx, batch_size=batch_size, seed=seed
        )
        print(data_loader)

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
                accept_frac = percentage_accepted(model=model, gen=gen)
                log_dict["accept_frac"] = accept_frac
                print(f"Epoch {epoch}: {loss.item()} {accept_frac=}")
            run.log(log_dict)
            if epoch > num_epochs:
                break
        return losses

# %%
train = True
from datetime import datetime

timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
model_file_name = f"model_weights_parity_{cfg.n_layers}l_{cfg.d_model}_{timestamp}.pt"

if train:
    losses = train_basic_model(
        model, gen, batch_size=32, num_epochs=20_000, seed=123
    )
    fig = px.line(losses, labels={"x": "Epoch", "y": "Loss"})
    fig.show()
    t.save(model.state_dict(), model_file_name)
else:
    model_file_name = "model_weights_parity_8l_192_2023_06_26-07_40_02_PM.pt"
    model.load_state_dict(t.load(model_file_name))

# %%


# prompt = t.zeros((1,20), dtype=t.int64)
# Generate 1000 completions and see how many of them are accepted by the DFA
print(
    f"Fraction of accepted completions: {percentage_accepted(t.zeros((1000,1), dtype=t.int64) + ord('B'))}"
)

tokens, states = gen.batches_and_states_gen(
    batch_size=1000, word_len=model.cfg.n_ctx
).__next__()
print(tokens.device)


# %%

# Training a linear probe to detect whether there are an even number of As so far

t.cuda.empty_cache()
# First gather data: create a dataset, compute true states, and use model.run_with_cache to get hidden states
with t.inference_mode():
    model.eval()
    logits, cache = model.run_with_cache(tokens)
    print(f"logits is shape {logits.shape} with total {logits.numel():_} elements")

# %%
# Now train a linear probe on states...
assert states[(states != 1) & (states != 2)].flatten().sum() == 0

# 0 = even, 1 = odd
states_transformed = states - 1
states_transformed = (
    rearrange(states_transformed[:, 1:], "b p -> (b p)").cpu().detach().numpy()
)


# %%
train_probes = False
if train_probes:
    for layer in range(model.cfg.n_layers):
        for loc in ("mid", "post"):
            resid = cache["resid_" + loc, layer]
            # Independent variable for the probe is the residual at just that position
            resid = rearrange(resid[:, :], "b p d_model -> (b p) d_model")
            resid = resid.cpu().detach().numpy()
            # Use logistic regression to predict whether there are an even number of As
            lr = LogisticRegression(max_iter=10000, solver="lbfgs")
            lr.fit(resid, states_transformed)

            score = lr.score(resid, states_transformed)
            print(f"Score for {layer} {loc}: {score}")
# %%

# Hypothesis: the model counts the number of As since the last C, then computes even/odd
# We can test this by replacing Cs with Bs and seeing if accuracy drops

# First get baseline performance
prompt = tokens[:, :16]
print(prompt.shape)
print(f"Baseline performance: {percentage_accepted(prompt)}")

tokens_noised = tokens.clone()
tokens_noised[tokens_noised == ord("C")] = ord("B")
prompt_noised = tokens_noised[:, :16]
print(f"Performance with all C replaced by B: {percentage_accepted(tokens_noised)}")
# accuracy seems to increase because the model now refuses to generate Cs
# maybe this means the model copies Cs?

# %%

dfa: DFA = gen.dfa
dfa.accepts_input("AABCCBBCBAABBCBAAABCAABAABBACCA")
list(dfa.read_input_stepwise("AABCCBBCBAABBCBAAABCAABAABBACCA", ignore_rejection=True))


def print_dfa_trajectory(dfa: DFA, s):
    """s is input string"""
    print(" " + s)
    states = dfa.read_input_stepwise(s, ignore_rejection=True)
    # Print states, highlighting rejection states in red
    for state in states:
        if state not in dfa.final_states:
            print(f"\033[91m{state}\033[0m", end="")
        else:
            print(state, end="")
    print()


# %%

# Exploring the model's errors
prompt = t.zeros((2000, 1), dtype=t.int64) + ord("A")
generated_tokens = model.generate(
    input=prompt, eos_token_id=0, max_new_tokens=31, do_sample=True, temperature=1
)
generated_strings = [gen.detokenize(line).strip("\x00") for line in generated_tokens]
for string in generated_strings:
    print_dfa_trajectory(dfa, string)
    print()

# %%
# Make histogram of first failure position
failure_positions = []
for string in generated_strings:
    states = list(dfa.read_input_stepwise(string, ignore_rejection=True))
    if states[-1] not in dfa.final_states:
        failure_positions.append(states.index(states[-1]))

fig = px.histogram(failure_positions)
fig.show()
# %%

# Callum's graphing functions

update_layout_set = {
    "xaxis_range",
    "yaxis_range",
    "hovermode",
    "xaxis_title",
    "yaxis_title",
    "colorbar",
    "colorscale",
    "coloraxis",
    "title_x",
    "bargap",
    "bargroupgap",
    "xaxis_tickformat",
    "yaxis_tickformat",
    "title_y",
    "legend_title_text",
    "xaxis_showgrid",
    "xaxis_gridwidth",
    "xaxis_gridcolor",
    "yaxis_showgrid",
    "yaxis_gridwidth",
    "yaxis_gridcolor",
    "showlegend",
    "xaxis_tickmode",
    "yaxis_tickmode",
    "xaxis_tickangle",
    "yaxis_tickangle",
    "margin",
    "xaxis_visible",
    "yaxis_visible",
    "bargap",
    "bargroupgap",
}


def imshow(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(
        transformer_lens.utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        **kwargs_pre,
    )
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]["text"] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    # things like `xaxis_tickmode` should be applied to all subplots. This is super janky lol but I'm under time pressure
    for setting in ["tickangle"]:
        if f"xaxis_{setting}" in kwargs_post:
            i = 2
            while f"xaxis{i}" in fig["layout"]:
                kwargs_post[f"xaxis{i}_{setting}"] = kwargs_post[f"xaxis_{setting}"]
                i += 1
    fig.update_layout(**kwargs_post)
    fig.show(renderer=renderer)


def hist(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    names = kwargs_pre.pop("names", None)
    if "barmode" not in kwargs_post:
        kwargs_post["barmode"] = "overlay"
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.histogram(x=tensor, **kwargs_pre).update_layout(**kwargs_post)
    if names is not None:
        for i in range(len(fig.data)):
            fig.data[i]["name"] = names[i // 2]
    fig.show(renderer)


# %%


# %%
def show_heads(model, word: str, layer: Optional[int] = None):
    with t.inference_mode():
        model.eval()
        logits, cache = model.run_with_cache(gen.tokenize(word))
    layers = [layer] if layer is not None else range(model.cfg.n_layers)
    for layer in layers:
        attn = cache["pattern", layer][0].squeeze(0)
        print(f"layer {layer}")
        display(
            circuitsvis.attention.attention_heads(tokens=list(word), attention=attn)
        )


show_heads(model, "BBBBBBABBBABBB")
# %% [markdown]

## Notes for string *BBBBBBABBBABBB*
### Layer 0:
# `l0h0` pays attention to As.  With two A's it appears to be roughly equal
# Other heads appear to be uninteresting

### Layer 1:
# `l1h3` pays attention to the latest A only (But note with Cs)
# (mino) `l1h1` kinda pays attention to the latest A only

### Layer 2:
# `l1h3` pays attention to the latest A only
# (mino) `l1h1` kinda pays attention to the latest A only
# %%

# show_heads(model, "BBBCBBABBBABBBCBBBA")
# %%
show_heads(model, "BBABBABBABBABBABBABB")
# %% [markdown]
# # For * "BBABBABBABBABBABBABB"*
# Layer 7 head 1 is odd regions paying attention to odd regions and even regions paying attention to even regions?
#  One big exception to this for tokens after the last A.
# Layer 7 other heads basically pay attention to only tokens at even regions, with, again execption after the last A
# %%
