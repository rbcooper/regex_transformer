# %%
import tqdm
from transformer_lens import HookedTransformer
from torch.utils.data import Dataset, DataLoader
import model_training
import rubiks_datasets
import test_rubiks_dataset

# %%


def model_generates_valid_samples(
        dataset: Dataset, training_samples=1_000, samples_to_validate=30, max_new_tokens=40):
    training_args = {
        "training_samples": training_samples,
    }
    model = model_training.make_basic_model()
    model_training.train_model(model, dataset=dataset, dry_run=True, training_args=training_args)
    valids = []
    for _ in range(samples_to_validate):
        sample = model.generate(max_new_tokens=max_new_tokens)
        valids.append(rubiks_datasets.is_valid_sequence(sample))
    return all(valids)

def no_tests_no_moves_trains_easilly():
    length = 40
    dataset = rubiks_datasets.make_start_cube_dataset(length=length)
    is_valid = model_generates_valid_samples(dataset, max_new_tokens=length, training_samples=1_000)



def no_test_thing():
    dataset = rubiks_datasets.make_start_cube_dataset(length=11)
    num_samples = 1024
    batch_size = 8
    num_batches = num_samples // batch_size
    data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
        )
    for epoch, tokens in tqdm.auto.tqdm(
            enumerate(data_loader), total=num_batches
        ):
        print(tokens)


# %%
