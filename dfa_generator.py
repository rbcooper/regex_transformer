# %%
from automata.fa.dfa import DFA
from automata.fa.fa import FA
from automata.fa.nfa import NFA
from graphviz import Digraph
from IPython.display import Image, display
import torch as t
from typing import List, Tuple
from dataclasses import dataclass

# %%

def dfa_from_regex(regex: str):
    """
    Create a DFA from a regular expression
    """
    nfa = NFA.from_regex(regex)
    dfa = DFA.from_nfa(nfa)
    return dfa


def display_fa(automaton: FA):
    """
    Displays a finite automaton in a Jupyter Notebook
    """
    pdot = automaton.show_diagram()
    plt = Image(pdot.create_png())
    display(plt)

# %%

@dataclass
class DfaGenerator():
    dfa: DFA

    def get_batches_and_states(self, word_len: int, batch_size: int, seed: int = None) -> Tuple[t.Tensor, t.Tensor]:
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
