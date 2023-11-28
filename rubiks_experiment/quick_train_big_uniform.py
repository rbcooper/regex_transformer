# %%
import torch
import rubiks_datasets
import model_training
from transformer_lens import HookedTransformer, HookedTransformerConfig

# %%
model_config_dict = {
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
    "initializer_range": 0.05,
    "model_name": "regex-tester",
    "n_ctx": 126,
    "n_devices": 1,
    "n_heads": 6,
    "n_layers": 8,
    "n_params": 6553600,
    "normalization_type": "LN",
    "original_architecture": "neel",
    "parallel_attn_mlp": False,
    "positional_embedding_type": "standard",
    "rotary_dim": None,
    "scale_attn_by_inverse_layer_idx": False,
    "seed": None,
    "tokenizer_name": None,
    "use_attn_result": False,
    "use_attn_scale": True,
    "use_hook_tokens": False,
    "use_local_attn": False,
    "use_split_qkv_input": False,
    "window_size": None,
}


source_path = "models/some_8_layer_uniform4.pt"


def save_model(model, source_path=source_path):
    torch.save(model.state_dict(), source_path)


def load_model(source_path=source_path):
    model = HookedTransformer(HookedTransformerConfig(**model_config_dict))
    model.load_state_dict(torch.load(source_path))
    model.tokenizer = rubiks_datasets.tokenizer
    model.to("cuda")
    return model


# %%


def q_train_big_uniform_model(model):
    model_training.train_model(
        model,
        dataset=rubiks_datasets.make_uniform_prob_dataset(length=126, seed=None),
        dry_run=False,
        training_args={"training_samples": 500_000_000, "n_workers": 8, "batch_size": 32},
        logging_args={"save_every": 100_000},
    )

if __name__ == "__main__":
    model = load_model()
    import cProfile
    q_train_big_uniform_model(model)
    save_model(model, "models/some_8_layer_uniform6.pt")

# %%
