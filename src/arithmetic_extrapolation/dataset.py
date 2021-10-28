import torch
import random
from torch.utils.data import Dataset


class anbn(Dataset):
    """Defines the formal language A^n B^n. """
    vocab = ["a", "b", "<EOS>", "<PAD>"]
    char_to_ix = {"a": 0, "b": 1, "<EOS>": 2, "<PAD>": 3}
    ix_to_char = {0: "a", 1: "b", 2: "<EOS>", 3: "<PAD>"}

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

    def generate_words(self):
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
    char_to_ix = {"a": 0, "b": 1, "<EOS>": 2, "<PAD>": 3}
    ix_to_char = {0: "a", 1: "b", 2: "<EOS>", 3: "<PAD>"}

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


class dyckn(Dataset):
    """Defines the formal language A^n B^n. """
    vocab = ["<PAD>", "a", "b", "<EOS>"]
    char_to_ix = {"<PAD>": 0, "a": 1, "b": 2, "<EOS>": 3}
    ix_to_char = {0: "<PAD>", 1: "a", 2: "b", 3: "<EOS>"}

    def __init__(self, n, max_len):
        self.n = n
        self.max_len = max_len

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

    def generate_words(self):
        words = ["a" * n + "b" * n for n in range(self.min_n, self.max_n)]
        random.shuffle(words)
        return words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        sample = (self.words[idx], self.targets[idx])
        return sample


class addition(Dataset):
    """Defines the formal language A^n B^n. """
    vocab = ["<PAD>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", "<EOS>"]
    char_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "+": 10, "=": 11, "<EOS>": 12, "<PAD>": 13}
    ix_to_char = {v: k for k, v in char_to_ix.items()}

    def __init__(self, min, max, sample=1.):
        self.min = min
        self.max = max
        self.sample = sample
        words, targets = self.generate_words()
        words = self.convert_to_ix(words)
        targets = self.convert_to_ix(targets, is_target=True)
        self.words = words
        self.targets = targets

    def convert_to_ix(self, inputs, is_target=False):
        as_ix = []
        for input in inputs:
            input_as_ix = []
            for c in input:
                input_as_ix.append(addition.char_to_ix[c])
            if is_target:
                input_as_ix.append(addition.char_to_ix["<EOS>"])
                y = torch.zeros(len(input_as_ix), len(addition.vocab))
                y[range(y.shape[0]), input_as_ix] = 1
                as_ix.append(y)
            else:
                as_ix.append(torch.LongTensor(input_as_ix))
        return as_ix

    def generate_words(self):
        samples = []
        for i1 in range(self.min, self.max):
            for i2 in range(self.min, self.max):
                samples.append(("".join([str(i1), "+", str(i2)]), str(i1+i2)))
        random.shuffle(samples)
        samples = random.sample(samples, int(self.sample * len(samples)))
        words = [sample[0] for sample in samples]
        targets = [sample[1] for sample in samples]
        return words, targets

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        sample = (self.words[idx], self.targets[idx])
        return sample
