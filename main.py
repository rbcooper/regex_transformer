# %%

import torch as t


import transformer_lens
from transformer_lens import HookedTransformer
import numpy as np
import random
import plotly.express as px
import tqdm

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
        "d_mlp": 512,
        "d_model": 128,
        "d_vocab": 256,
        "d_vocab_out": 256,
        "device": "cuda",
        "eps": 1e-05,
        "final_rms": False,
        "from_checkpoint": False,
        "gated_mlp": False,
        "init_mode": "gpt2",
        "init_weights": True,
        "initializer_range": 0.035355339059327376,
        "model_name": "GELU_4L512W_C4_Code",
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


def to_str(tokens):
    return "".join([chr(t) for t in tokens])


# %%

# Python list of 100 random English words
import nltk

nltk.download("words")
from nltk.corpus import words

all_words = words.words()
rng = np.random.default_rng(0)
np.random.shuffle(all_words)
words = all_words[:100]
print(words)

# %%


def make_data_generator(
    cfg: transformer_lens.HookedTransformerConfig, words, batch_size, seed=123
):
    rng = np.random.default_rng(seed)

    while True:
        batch = t.zeros((batch_size, cfg.n_ctx), dtype=t.long)
        for i in range(batch_size):
            word = rng.choice(words)
            batch[i, 1 : 1 + len(word)] = t.tensor(list(bytes(word, encoding="utf-8")))

        yield batch.to(device)  # (batch_size, seq_len)


data_generator = make_data_generator(cfg, words, 2)
data_generator.__next__()  # looks like words
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
tokens = data_generator.__next__()
logits = model(tokens)
loss_fn(logits, tokens)

# %%

# training the model


density = 0.02
# batch_size = min(256, hashtable_size(density, cfg.d_vocab, cfg.n_ctx))
batch_size = 32
print(f"Batch size: {batch_size}")
num_epochs = 4000
lr = 1e-4
betas = (0.9, 0.95)
max_grad_norm = 1.0
wd = 0.1
optimizer = t.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 100, 1.0))
data_loader = make_data_generator(cfg, words=words, batch_size=batch_size, seed=123)

n_parameters = sum(p.numel() for p in model.parameters())
n_grams_size = (
    np.log2(cfg.d_vocab - 1) * density * (cfg.d_vocab - 1) ** (cfg.n_ctx - 2) / 1e6
)
parameter_size = n_parameters * model.parameters().__next__().element_size() / 1e6
print(f"n-grams are {n_grams_size:.2f} MB long")
print(f"Model has {n_parameters} parameters = {parameter_size} MB")
print(f"Ratio of sizes: {n_grams_size / parameter_size:.4f}")

"""## Model Training"""

losses = []
for epoch in tqdm.tqdm(range(num_epochs)):
    tokens = next(data_loader)
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
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: {loss.item()}")

# %%
fig = px.line(losses, labels={"x": "Epoch", "y": "Loss"})
fig.show()

# %%
tokens = model.generate(input=t.tensor([[111]], dtype=t.int64), eos_token_id=0)
to_str(tokens[0])


# %%
