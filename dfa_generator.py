# %%
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch as t
from automata.fa.dfa import DFA
from automata.fa.fa import FA
from automata.fa.nfa import NFA

from frozendict import frozendict
from graphviz import Digraph
from IPython.display import display, Image

# %%


def dfa_from_regex(regex: str):
    """∏
    Create a DFA from a regular expression
    """
    nfa = NFA.from_regex(regex)
    dfa = DFA.from_nfa(nfa)
    dfa = make_start_state_zero(dfa)
    return dfa


def display_fa(automaton: FA):
    """
    Displays a finite automaton in a Jupyter Notebook
    """
    pdot = automaton.show_diagram()
    plt = Image(pdot.create_png())
    display(plt)


def make_start_state_zero(dfa: DFA):
    """ """
    if dfa.initial_state == 0:
        return dfa
    old_start_state = dfa.initial_state

    def rename_state(state):
        if state == old_start_state:
            return 0
        elif state == 0:
            return old_start_state
        else:
            return state

    def rename_transitions(state_transition: Dict[str, int]):
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
        """Returns a tuple of (batches, states).  The batches are a tensor of shape (batch_size, word_len)"""
        batches = t.zeros((batch_size, word_len), dtype=t.int32)
        states = t.zeros((batch_size, word_len + 1), dtype=t.int32)
        for i in range(batch_size):
            word = self.dfa.random_word(word_len, seed=seed)
            word_states = self.dfa.read_input_stepwise(word)
            batches[i, :] = t.tensor(list(int(w) for w in word))
            states[i, :] = t.tensor(list(int(s) for s in word_states))
        return batches, states

    def batches_and_states_gen(self, word_len: int, batch_size: int, seed: int = None):
        while True:
            yield self.get_batches_and_states(word_len, batch_size, seed=seed)

    @staticmethod
    def from_regex(regex: str):
        dfa = dfa_from_regex(regex)
        return DfaGenerator(dfa)


# %%
