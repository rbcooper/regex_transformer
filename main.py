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

cfg = transformer_lens.HookedTransformerConfig(**
{'act_fn': 'gelu',
 'attention_dir': 'causal',
 'attn_only': False,
 'attn_types': None,
 'checkpoint_index': None,
 'checkpoint_label_type': None,
 'checkpoint_value': None,
 'd_head': 32,
 'd_mlp': 512,
 'd_model': 128,
 'd_vocab': 256,
 'd_vocab_out': 256,
 'device': 'cuda',
 'eps': 1e-05,
 'final_rms': False,
 'from_checkpoint': False,
 'gated_mlp': False,
 'init_mode': 'gpt2',
 'init_weights': True,
 'initializer_range': 0.035355339059327376,
 'model_name': 'GELU_4L512W_C4_Code',
 'n_ctx': 32,
 'n_devices': 1,
 'n_heads': 4,
 'n_layers': 4,
 # 'n_params': 12582912,
 'normalization_type': 'LN',
 'original_architecture': 'neel',
 'parallel_attn_mlp': False,
 'positional_embedding_type': 'standard',
 'rotary_dim': None,
 'scale_attn_by_inverse_layer_idx': False,
 'seed': None,
#  'tokenizer_name': 'NeelNanda/gpt-neox-tokenizer-digits',
 'use_attn_result': False,
 'use_attn_scale': True,
 'use_hook_tokens': False,
 'use_local_attn': False,
 'use_split_qkv_input': False,
 'window_size': None}
)

# %%
cfg.n_params
# %%

model = HookedTransformer(cfg)

# %%

def to_str(tokens):
    """
    Converts a list of character values to a string.
    """
    return "".join([chr(t) for t in tokens])

# %%

# Python list of 100 random English words
import nltk
nltk.download('words')
from nltk.corpus import words
all_words = words.words()
rng = np.random.default_rng(0)
np.random.shuffle(all_words)
words = all_words[:100]
print(words)

# %%

def make_data_generator(cfg:transformer_lens.HookedTransformerConfig, words, batch_size, seed=123):
    rng = np.random.default_rng(seed)

    while True:
        batch = t.zeros((batch_size, cfg.n_ctx), dtype=t.long)
        for i in range(batch_size):
            word = rng.choice(words)
            batch[i, 1:1+len(word)] = t.tensor(list(bytes(word, encoding="utf-8")))
            
        yield batch.to(device) # (batch_size, seq_len)

data_generator = make_data_generator(cfg, words, 2)
data_generator.__next__() # looks like words
# %%

def loss_fn(logits, tokens, return_per_token=False):
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    # assert False
    logits = logits[:, :-1] # ignore last logit
    tokens = tokens[:, 1:] # ignore first token (bos token)
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    loss = -correct_log_probs.mean() # mean over batch and pos
    return loss

# %%
tokens = data_generator.__next__()
logits = model(tokens)
loss_fn(logits, tokens)

# %%

# training the model


density = 0.02
#batch_size = min(256, hashtable_size(density, cfg.d_vocab, cfg.n_ctx))
batch_size = 32
print(f"Batch size: {batch_size}")
num_epochs = 4000
lr = 1e-4
betas = (0.9, 0.95)
max_grad_norm = 1.0
wd = 0.1
optimizer = t.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i/100, 1.))
data_loader = make_data_generator(cfg, words=words, batch_size=batch_size, seed=123)

n_parameters = sum(p.numel() for p in model.parameters())
n_grams_size = np.log2(cfg.d_vocab - 1) * density * (cfg.d_vocab - 1) ** (cfg.n_ctx - 2) / 1e6
parameter_size = n_parameters * model.parameters().__next__().element_size() / 1e6
print(f"Model has {n_parameters} parameters = {parameter_size} MB")

"""## Model Training"""

losses = []
for epoch in tqdm.auto.tqdm(range(num_epochs)):
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
    # if epoch % 100 == 0:
    #     print(f'Epoch {epoch}: {loss.item()}')

# %%
fig = px.line(losses, labels={"x":"Epoch", "y":"Loss"})
fig.show()

# %%
tokens = model.generate(input=t.tensor([[0, 111]], dtype=t.int64), eos_token_id=0, max_new_tokens=50)
to_str(tokens[0])


# %%

from dataclasses import dataclass


def aa_generator(seed=123, batch_size=32, n_ctx=32) -> t.Tensor:
    rng = np.random.default_rng(seed)
    while True:
        length = rng.geometric(0.5, size=(batch_size,)) * 2
        length = np.clip(length, 2, n_ctx - 1)
        batch = t.zeros((batch_size, n_ctx), dtype=t.long)
        for i in range(batch_size):
            batch[i, 1:1+length[i]] = ord('A')
        yield batch.to(device)

def even_As_generator(seed=123, batch_size=32, n_ctx=32) -> t.Tensor:
    """
    Defines matrices that correspond to generator for string with even As,
    then returns that generator
    """
    rng = np.random.default_rng(seed)
    while True:
        batch = []
        for i in range(batch_size):
            result = ""
            state = "S0"
            while True:
                if len(result) >= n_ctx - 1:
                    batch.append(result)
                    break
                if state == "S0":
                    if rng.random() < 0.2:
                        # result += "\x00"
                        batch.append(result)
                        break
                    elif rng.random() < 0.5:
                        result += "A"
                        state = "S1"
                    else:
                        result += "B"
                        state = "S0"
                elif state == "S1":
                    if rng.random() < 0.5:
                        result += "A"
                        state = "S0"
                    else:
                        result += "B"
                        state = "S1"

        ret = t.zeros((batch_size, n_ctx), dtype=t.long)
        for i, s in enumerate(batch):
            ret[i, 1:1+len(s)] = t.tensor(list(bytes(s, encoding="utf-8")))
        yield ret.to(device)

test_tensor = even_As_generator(n_ctx=10, seed=1).__next__()

for line in test_tensor:
    # lines that contain a second 0 must have an even number of As
    assert sum(line == 65) % 2 == 0 or sum(line == 0) == 1

# %%
# aa_train = make_aa_generator(seed=123)
# aa_test = make_aa_generator(seed=456)

def train_basic_model(model, gen, batch_size=100, seed=123):
    print(f"Batch size: {batch_size}")
    num_epochs = 20000
    lr = 1e-4
    betas = (0.9, 0.95)
    max_grad_norm = 1.0
    wd = 0.1
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i/100, 1.))
    data_loader = gen(seed=seed,batch_size=batch_size, n_ctx=model.cfg.n_ctx)
    print(data_loader)

    n_parameters = sum(p.numel() for p in model.parameters())
    parameter_size = n_parameters * model.parameters().__next__().element_size() / 1e6
    print(f"Model has {n_parameters} parameters = {parameter_size} MB")

    """## Model Training"""

    losses = []
    for epoch in tqdm.auto.tqdm(range(num_epochs)):
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
            print(f'Epoch {epoch}: {loss.item()}')
    return losses

model = HookedTransformer(cfg)
losses = train_basic_model(model, even_As_generator, batch_size=32, seed=123)
# %%
fig = px.line(losses, labels={"x":"Epoch", "y":"Loss"})
fig.show()


# %%

# prompt = t.zeros((1,20), dtype=t.int64)
# Generate 1000 completions and see how many of them have an even number of As
generated_tokens = model.generate(input=t.zeros((1000,1), dtype=t.int64),
                        eos_token_id=0, max_new_tokens=31,
                        do_sample=True, temperature=1)
even_As = [(sum(line == 65) % 2 == 0).item() for line in generated_tokens]
print(f"Fraction of completions with even number of As: {sum(even_As)/len(even_As)}")

generated_strings = [to_str(line).strip('\x00') for line in generated_tokens]
[x for x in zip(generated_strings, even_As)]

# Now make histogram of length, with color = whether it has even number of As
lengths = [len(s) for s in generated_strings]
fig = px.histogram(lengths, color=even_As, labels={"value":"Length", "color":"Even number of As"})
fig.show()

# %%

from tokenizers import Tokenizer
from tokenizers.models import CharacterLevel

tokenizer = Tokenizer(CharacterLevel())


# %%


tokens = even_As_generator(batch_size=10000, n_ctx=model.cfg.n_ctx).__next__()

even_As_proportion = ((tokens == ord('A')).sum(axis=1) % 2 == 0).sum() / len(tokens)
print(f"Fraction of even As in true dataset: {even_As_proportion}")
# %%
