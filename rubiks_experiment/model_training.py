# %%
from datetime import datetime
from frozendict import frozendict
import torch
import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
from torch.utils.data import Dataset, DataLoader
import wandb
import rubiks_datasets


base_model_config = frozendict(
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

base_training_args = frozendict(
    batch_size=128,
    lr=1e-4,
    betas=(0.9, 0.95),
    max_grad_norm=1.0,
    weight_decay=0.01,
    num_workers=30,
    training_samples=100_000,
)

base_logging_args = frozendict(
    project="rubiks_color_query_uniform",
    entity="alighnment",
    log_every=500,
    save_every=10_000,
    tags=["colorquery", "uniform", "rubiks"],
)

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
    # ignore everything but colors
    mask = torch.isin(tokens, rubiks_datasets.color_ids)
    print(f"{logits.shape=}, {tokens.shape=}")
    # print(rubiks_generator.tokenizer.decode(list(tokens[0])))
    log_probs = logits.log_softmax(-1)
    print(f"{log_probs.shape=}, {tokens.shape=}")
    logits = logits[mask, ...]
    tokens = tokens[mask, ...]
    print(f"{logits.shape=}, {tokens.shape=}")
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    loss = -correct_log_probs.mean()  # mean over batch and pos
    return loss


logit_metrics = {
    #    "color_loss": color_loss,
    # Color loss
    # portion valid
    # average first invalid
}

model_metrics = {
    # "generate_sample": lambda m: m.generate(max_new_tokens=40, verbose=False),
    "corrected_sample": lambda m: rubiks_datasets.pretty_incorrect_colors(
        m.generate(max_new_tokens=120, verbose=False)
    ),
}

# %%


def make_basic_model(model_config={}):
    model_config = {**base_model_config, **model_config}
    cfg = HookedTransformerConfig(**model_config)
    model = HookedTransformer(cfg)
    # model.tokenizer = rubiks_datasets.tokenizer
    model.tokenizer = rubiks_datasets.tokenizer
    return model


def train_model(
    model: HookedTransformer,
    dataset: Dataset,
    training_args={},
    logging_args={},
    loss_fn=loss_fn,
    dry_run: bool = True,
) -> None:
    training_args = {**base_training_args, **training_args}
    logging_args = {**base_logging_args, **logging_args}
    with wandb.init(
        project=logging_args["project"],
        entity=logging_args["entity"],
        job_type="train",
        tags=logging_args["tags"],
        mode="offline" if dry_run else "online",
    ) as run:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args["lr"],
            betas=training_args["betas"],
            weight_decay=training_args["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda i: min(i / 100, 1.0)
        )

        batch_size = training_args["batch_size"]
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=training_args["num_workers"],
        )

        n_parameters = sum(p.numel() for p in model.parameters())
        parameter_size = (
            n_parameters * model.parameters().__next__().element_size() / 1e6
        )
        print(f"Training with config: \n{model.cfg}")
        print(f"Model has {n_parameters} parameters = {parameter_size} MB")

        """## Model Training"""
        model.train()
        training_samples = training_args["training_samples"]
        num_batches = training_samples // batch_size
        sample_num = 0
        for batch_num, tokens in tqdm.auto.tqdm(
            enumerate(data_loader), total=num_batches
        ):
            sample_num += len(tokens)
            tokens = tokens.cuda()
            logits = model(tokens)
            loss = loss_fn(logits, tokens)
            loss.backward()
            max_grad_norm = training_args["max_grad_norm"]
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            log_dict = dict(
                loss=loss.item(),
            )
            if batch_num % logging_args["log_every"] == 0:
                for metric_name, func in logit_metrics.items():
                    log_dict[metric_name] = func(logits, tokens)
                for metric_name, func in model_metrics.items():
                    log_dict[metric_name] = func(model)
                log_dict_pretty = " ".join(
                    (f"{k}={v:.5f}" if isinstance(v, (float, int)) else f"{k}='{v}'")
                    for k, v in log_dict.items()
                )
                print(log_dict_pretty)
            save_every = logging_args["save_every"]
            if save_every is not None and batch_num % save_every == 0 and batch_num > 0:
                dataset_name = getattr(dataset, "name", "some_dataset")
                timestamp = datetime.now().strftime("%Y_%m_%d_%I_%M_%S")
                model_file_name = f"models/checkpoint_rubiks_{dataset_name}_{model.cfg.n_layers}l_{model.cfg.d_model}d_{model.cfg.n_ctx}c_{sample_num}_{timestamp}.pt"
                torch.save(model.state_dict(), model_file_name)
                wandb.save(model_file_name)

            run.log(log_dict, step=sample_num)
            if sample_num > training_samples:
                break


# %%


def model_generates_valid_samples(
    dataset: Dataset,
    training_samples=500_000,
    samples_to_validate=30,
    max_new_tokens=40,
):
    training_args = {
        "training_samples": training_samples,
    }
    model = make_basic_model()
    train_model(model, dataset=dataset, dry_run=True, training_args=training_args)
    print("Training done")
    valids = []
    for _ in range(samples_to_validate):
        sample = model.generate(max_new_tokens=max_new_tokens)
        valids.append(rubiks_datasets.is_valid_sequence(sample))
    print(valids)
    return all(valids)


def tests_no_moves_trains_easilly():
    length = 126
    dataset = rubiks_datasets.make_start_cube_dataset(length=length)
    is_valid = model_generates_valid_samples(
        dataset, max_new_tokens=length, training_samples=50_000
    )


# tests_no_moves_trains_easilly()
# %%
