import torch
import random
from torch.utils.data import Dataset


class anbn(Dataset):
    """Defines the formal language A^n B^n. """
    vocab = ["<PAD>", "a", "b", "<EOS>"]
    char_to_ix = {"<PAD>": 0, "a": 1, "b": 2, "<EOS>": 3}
    ix_to_char = {0: "<PAD>", 1: "a", 2: "b", 3: "<EOS>"}

    def __init__(self, min_n, max_n):
        self.min_n = min_n
        self.max_n = max_n

        words = self.generate_words()
        words = self.convert_words_to_ix(words)
        self.words = words
        self.targets = self.construct_targets(words)

    def construct_targets(self, words):
        targets = []
        for word in words:
            target = word.detach().clone()
            target[-1] = anbn.char_to_ix["<EOS>"]
            y = torch.zeros(target.shape[0], len(anbn.vocab))
            y[range(y.shape[0]), target] = 1  # set one hots like words
            targets.append(y)
        return targets

    def convert_words_to_ix(self, words):
        words_as_ix = []
        for widx, word in enumerate(words):
            word_as_ix = []
            for cidx, c in enumerate(word):
                word_as_ix.append(anbn.char_to_ix[c])
            words_as_ix.append(torch.LongTensor(word_as_ix))
        return words_as_ix

    def generate_words(self, shuffle=True):
        words = ["a" * n + "b" * n for n in range(self.min_n, self.max_n)]
        random.shuffle(words)
        return words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        sample = (self.words[idx], self.targets[idx])
        return sample


class anbnEval(Dataset):
    """Defines the formal language A^n B^n. """
    vocab = ["<PAD>", "a", "b", "<EOS>"]
    char_to_ix = {"<PAD>": 0, "a": 1, "b": 2, "<EOS>": 3}
    ix_to_char = {0: "<PAD>", 1: "a", 2: "b", 3: "<EOS>"}

    def __init__(self, word):
        words = [word]
        words = self.convert_words_to_ix(words)
        self.words = words
        self.targets = self.construct_targets(words)

    def construct_targets(self, words):
        targets = []
        for word in words:
            target = word.detach().clone()
            target[-1] = anbn.char_to_ix["<EOS>"]
            y = torch.zeros(target.shape[0], len(anbn.vocab))
            y[range(y.shape[0]), target] = 1  # set one hots like words
            targets.append(y)
        return targets

    def convert_words_to_ix(self, words):
        words_as_ix = []
        for widx, word in enumerate(words):
            word_as_ix = []
            for cidx, c in enumerate(word):
                word_as_ix.append(anbn.char_to_ix[c])
            words_as_ix.append(torch.LongTensor(word_as_ix))
        return words_as_ix

    def generate_words(self, shuffle=True):
        words = ["a" * n + "b" * n for n in range(self.min_n, self.max_n)]
        random.shuffle(words)
        return words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        sample = (self.words[idx], self.targets[idx])
        return sample
