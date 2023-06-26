# %%
from torch.utils.data import Dataset

from dataclasses import dataclass
from typing import Dict, Generator, Tuple

import torch as t
from automata.fa.dfa import DFA
from automata.fa.fa import FA
from automata.fa.nfa import NFA

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
    def from_regex(regex: str) -> DFA:
        dfa = dfa_from_regex(regex)
        return DfaGenerator(dfa)

    def tokenize(self, word: str) -> t.Tensor:
        return t.tensor(list(ord(w) for w in word))

    def detokenize(self, tensor: t.Tensor) -> str:
        if len(tensor.shape) == 1:
            return "".join(chr(c) for c in tensor)
        elif len(tensor.shape) == 2:
            return [self.detokenize(w) for w in tensor]

    def display_fa(self):
        """Displays a finite automaton in a Jupyter Notebook"""
        display_fa(self.dfa)

    def dataset(self, length=20) -> Dataset:
        return DfaDataset(self, length)


class DfaDataset(Dataset):
    def __init__(self, dfa_gen: DfaGenerator, static_length: int):
        self.dfa_gen = dfa_gen
        self.dfa = dfa_gen.dfa
        self.word_len = static_length

    def __len__(self):
        return (2**31) - 1

    def __getitem__(self, idx):
        seed = idx
        word = self.dfa.random_word(k=self.word_len, seed=seed)
        word_states = self.dfa.read_input_stepwise(word)
        # TODO switch to constructing dataset / dataloader
        batches = t.tensor(list(ord(w) for w in word))
        states = t.tensor(list(int(s) for s in word_states))
        return batches, states


# %%
C_IF_EVEN_AS_DFA_GEN = DfaGenerator.from_regex("((B|C)*AB*A)*(B|C)*")


# %%
