# %%
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple, Union
import functools
import hashlib

import torch as t
from automata.fa.dfa import DFA
from automata.fa.fa import FA
from automata.fa.nfa import NFA
import rich

from frozendict import frozendict
from IPython.display import display, Image

# %%


def dfa_from_regex(regex: str) -> DFA:
    """
    Create a DFA from a regular expression
    """
    nfa = NFA.from_regex(regex)
    dfa = DFA.from_nfa(nfa)
    # dfa = make_start_state_zero(dfa)
    return dfa


def display_fa(automaton: FA) -> None:
    """
    Displays a finite automaton in a Jupyter Notebook
    """
    pdot = automaton.show_diagram()
    plt = Image(pdot.create_png())
    display(plt)


def make_start_state_zero(dfa: DFA) -> DFA:
    """ """
    if dfa.initial_state == 0:
        return dfa
    old_start_state = dfa.initial_state

    def rename_state(state: int) -> int:
        if state == old_start_state:
            return 0
        elif state == 0:
            return old_start_state
        else:
            return state

    def rename_transitions(state_transition: Dict[str, int]) -> Dict[str, int]:
        """
        renames the states in the transition from a single state.  This isn't
        the complete transition function, only the transition function from a
        given state.
        """
        result = {}
        for state, transition in state_transition.items():
            result[state] = frozendict(
                (c, rename_state(s)) for c, s in sorted(transition.items())
            )
        return frozendict(sorted(result.items()))

    states = frozenset(sorted(dfa.states))
    input_symbols = frozenset(sorted(dfa.input_symbols))
    final_states = frozenset(rename_state(s) for s in dfa.final_states)
    initial_state = rename_state(dfa.initial_state)

    transitions = rename_transitions(dfa.transitions)
    return DFA(
        states=states,
        input_symbols=input_symbols,
        transitions=transitions,
        initial_state=initial_state,
        final_states=final_states,
    )


# %%


@dataclass
class DfaGenerator:
    dfa: DFA

    def get_batches_and_states(
        self, word_len: int, batch_size: int, seed: int = None
    ) -> Tuple[t.Tensor, t.Tensor]:
        """
        Returns a tuple of (batches, states).  The batches are a tensor of shape (batch_size, word_len)
        word_len should be the model's n_ctx∏
        """
        batches = t.zeros((batch_size, word_len), dtype=t.int64)
        states = t.zeros((batch_size, word_len + 1), dtype=t.int64)
        for i in range(batch_size):
            word = self.dfa.random_word(word_len, seed=seed)
            word_states = self.dfa.read_input_stepwise(word)
            # TODO switch to constructing dataset / dataloader
            batches[i, :] = t.tensor(list(ord(w) for w in word))
            states[i, :] = t.tensor(list(int(s) for s in word_states))
        return batches, states

    def batches_and_states_gen(
        self, word_len: int, batch_size: int, seed: int = None
    ) -> Generator[Tuple[t.Tensor, t.Tensor], None, None]:
        while True:
            yield self.get_batches_and_states(word_len, batch_size, seed=seed)

    @staticmethod
    def from_regex(regex: str):
        dfa = dfa_from_regex(regex)
        return DfaGenerator(dfa)

    def tokenize(self, word: str) -> t.Tensor:
        return t.tensor(list(ord(w) for w in word))

    def detokenize(self, tensor: t.Tensor) -> str:
        if len(tensor.shape) == 1:
            return "".join(chr(c) for c in tensor)
        elif len(tensor.shape) == 2:
            return [self.detokenize(w) for w in tensor]

    def _tensorize_tokens_function(self, func, dtype):
        @functools.wraps(func)
        def fun(s: Union[str, t.Tensor]):
            if isinstance(s, str):
                return func(s)
            elif isinstance(s, t.Tensor):
                if s.dim() == 1:
                    return func(self.detokenize(s))
                elif s.dim() == 2:
                    batch, length = s.shape
                    result = t.zeros((batch), dtype=dtype, device=s.device)
                    for i in range(batch):
                        result[i] = func(self.detokenize(s[i]))
                    return result
            else:
                raise ValueError(f"Unknown input type: {type(s)}")

        return fun

    def display_fa(self) -> None:
        """Displays a finite automaton in a Jupyter Notebook"""
        display_fa(self.dfa)

    def dataset(self, word_length: int, seed=None) -> Dataset:
        """A torch dataset yielding (random_word, dfa_state) pairs)
        dfa_state is sequence length + 1, because the initial state is included
        """
        return DfaDataset(self, word_length, seed=seed)

    def dataloader(self, word_length: int, batch_size: int, seed=None) -> DataLoader:
        return DataLoader(self.dataset(word_length), batch_size=batch_size)

    def accepts(self, chars: Union[str, t.Tensor]) -> Union[bool, t.Tensor]:
        """Returns true if the dfa accepts the given string.
        If the input is a 1-d tensor, detokenize it first.  If the input is a 2-d
        tensor, apply accepts to each row and return a 1-d tensor of bools."""

        def accept(chars: str):
            return self.dfa.accepts_input(chars)

        return self._tensorize_tokens_function(accept, t.bool)(chars)

    def random_word(self, length: int, seed: int = None) -> str:
        return self.dfa.random_word(length, seed=seed)

    def pprint_dfa_trajectory(self, s: Union[str, t.Tensor]) -> None:
        """"""
        line1 = " " + s
        states = self.dfa.read_input_stepwise(s, ignore_rejection=True)
        # Print states, highlighting rejection states in red
        state_parts = []
        for state in states:
            if state in self.dfa.final_states:
                state_parts.append(f"[green]{state}[/green]")
            else:
                if state is None:
                    state = "�"
                state_parts.append(f"[red]{state}[/red]")
        line2 = "".join(state_parts)
        rich.print("\n".join([line1, line2]))


class DfaDataset(Dataset):
    def __init__(
        self, dfa_gen: DfaGenerator, static_length: int, *, seed: Optional[int] = None
    ):
        self.dfa_gen = dfa_gen
        self.dfa = dfa_gen.dfa
        self.word_len = static_length
        self.seed = seed

    def _munge_number(self, idx: int, seed: int) -> int:
        if self.seed is None:
            return idx
        idx ^= 1494354063
        seed ^= 882_789_810
        return (idx * seed) % (2**31)

    def __len__(self):
        return (2**31) - 1

    def __getitem__(self, idx):
        word = self.dfa.random_word(
            k=self.word_len, seed=self._munge_number(self.seed, idx)
        )
        word_states = self.dfa.read_input_stepwise(word, ignore_rejection=True)
        batches = t.tensor(list(ord(w) for w in word))
        states = t.tensor(list(int(s) for s in word_states))
        return batches, states


# %%
C_IF_EVEN_AS_DFA_GEN = DfaGenerator.from_regex("((B|C)*AB*A)*(B|C)*A?B*")


# %%
