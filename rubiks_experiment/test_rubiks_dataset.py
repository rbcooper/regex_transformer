from typing import List
import rubiks_datasets
import pycuber
from collections import defaultdict
from torch.utils.data import Dataset


def test_right_number_of_colors():
    all_pos = rubiks_datasets.all_sticker_positions_tokens
    assert len(all_pos) == 3 * 3 * 6
    assert len(rubiks_datasets.color_ids) == 6
    assert len(rubiks_datasets.all_move_ids) == 6 * 3
    assert len(rubiks_datasets.all_sticker_position_ids) == 3 * 3 * 6


def test_start_cube():
    assert_color_counts(rubiks_datasets.SOLVED_CUBE)


def assert_color_counts(cube: pycuber.Cube):
    color_to_pos = defaultdict(list)
    for pos in rubiks_datasets.all_sticker_positions_tokens:
        color_to_pos[rubiks_datasets.color_at(cube, pos)].append(pos)
    assert len(color_to_pos) == 6
    for color, positions in color_to_pos.items():
        assert len(positions) == 9
    assert set(color_to_pos) == set(rubiks_datasets.cube_colors)


def assert_good_dataset_is_good(dataset: Dataset):
    for i in range(10):
        sample = dataset[i]
        assert sample[0] == rubiks_datasets.tokenizer.bos_token_id
        made_valid = rubiks_datasets.with_correct_colors_after_queries(
            sample, truncate=True
        )
        made_valid = rubiks_datasets.ensure_token_list(made_valid)
        made_valid = made_valid[: len(sample)]
        assert rubiks_datasets.ensure_token_list(sample) == made_valid
        assert rubiks_datasets.is_valid_sequence(sample)


def make_some_datasets(length: int) -> List[Dataset]:
    datasets = [
        rubiks_datasets.make_start_cube_dataset(length=length),
        rubiks_datasets.make_prob_query_dataset(length=length, query_prob=0.2),
        rubiks_datasets.make_prob_query_dataset(length=length, query_prob=0.8),
        rubiks_datasets.make_prob_query_dataset(length=length, query_prob=0.0),
        rubiks_datasets.make_prob_query_dataset(length=length, query_prob=1.0),
        rubiks_datasets.make_only_moves_dataset(length=length),
        rubiks_datasets.make_n_moves_then_query_dataset(length=length, n_moves=1),
        rubiks_datasets.make_n_moves_then_query_dataset(
            length=length, n_moves=length // 2
        ),
        rubiks_datasets.make_n_moves_then_query_dataset(
            length=length, n_moves=length - 1
        ),
        rubiks_datasets.make_uniform_prob_dataset(length=length),
    ]
    return datasets


def test_datasets_are_good():
    length = 20
    datasets = make_some_datasets(length)
    for dataset in datasets:
        assert_good_dataset_is_good(dataset)


def test_make_n_moves_then_query_dataset():
    for length in [1, 2, 5, 10]:
        for n_moves in 0, 1, 2, 5, 9:
            n_moves = min(n_moves, length - 1)
            dataset = rubiks_datasets.make_n_moves_then_query_dataset(
                length=length, n_moves=n_moves
            )
            for i in range(10):
                sample = dataset[i]
                assert sample[0] == rubiks_datasets.tokenizer.bos_token_id
                assert len(sample) == length
                assert (
                    len([m for m in sample if m in rubiks_datasets.all_move_ids])
                    == n_moves
                )

if __name__ == "__main__":
    for d in make_some_datasets(20):
        rubiks_datasets.print_dataset_info(d, n=10)
        print()

