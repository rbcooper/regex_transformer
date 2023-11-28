# %%

import importlib
from datetime import datetime
import math
import training
import rubiks_datasets
import torch as t

from transformer_lens import HookedTransformer

for module in [training, rubiks_datasets]:
    importlib.reload(training)

# %%


def train_all_corner_aligator(samples=5_000, batch_size=128) -> HookedTransformer:
    experiment_name = "move_query_color_poisson"
    print(f"Experiment name: {experiment_name}")
    model_cfg = training.cfg
    model = HookedTransformer(model_cfg)
    model.tokenizer = rubiks_generator.tokenizer
    train_data_generator = rubiks_generator.generate_2x2x2_move_query_color_poisson

    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    model_file_name = f"models/model_weights_rubiks_{model_cfg.n_layers}l_{model_cfg.d_model}_{timestamp}_{experiment_name}.pt"
    batch_size = 128
    epochs = math.ceil(samples / batch_size)

    losses = training.train_basic_model(
        model,
        batch_size=batch_size,
        num_epochs=epochs,
        seed=123,
        save_every=10_000,
        data_generator=train_data_generator,
        tags=[experiment_name],
    )
    t.save(model.state_dict(), model_file_name)
    return model


# %%
train = True
if train:
    model = train_all_corner_aligator()
    model.set_tokenizer(rubiks_generator.tokenizer)
else:
    model = HookedTransformer(training.cfg)
    model.load_state_dict(
        t.load(
            "models/model_weights_rubiks_6l_256_2023_07_23-10_01_01_AM_move_freq_uniform_02_to_09.pt"
        )
    )
    model.set_tokenizer(rubiks_generator.tokenizer)

# %%
model.to_tokens("UR'")
# %%
model("U2+++y")
# %%
