import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, vocab_size, device, bias=True, use_embedding=False,
                 set_linear_bias=False, linear_normal=False, linear_uniform=False, use_ln=False, ln_preact=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.use_embedding = use_embedding
        self.use_ln = use_ln
        self.ln_preact = ln_preact
        self.linear_normal = linear_normal
        self.linear_uniform = linear_uniform
        self.set_linear_bias = set_linear_bias
        self.device = device
        self.target_pad_idx = 3
        self.bias = bias
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, self.vocab_size)
        if self.set_linear_bias:
            self.linear._parameters['bias'] = Variable(torch.tensor([.48, .48, .04, 0]), requires_grad=True)
        if self.linear_uniform:
            torch.nn.init.xavier_uniform_(self.linear.weight)
        elif self.linear_normal:
            torch.nn.init.xavier_normal_(self.linear._parameters['weight'], gain=1.0)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=1,
            padding_idx=3
        )

    def init_hidden(self):
        h_0 = torch.zeros((1, self.batch_size, self.hidden_size))
        c_0 = torch.zeros((1, self.batch_size, self.hidden_size))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        h_0 = Variable(h_0)
        c_0 = Variable(c_0)
        return (h_0, c_0)

    def forward_save(self, x):
        self.hidden = self.init_hidden()
        batch_size, seq_len = x.size()
        x = x.unsqueeze(2).float()

        outs = []
        cs = []
        hs = []
        # params = self.get_params()

        for cidx in range(x.size(1)):
            # [input_gate | forget_gate | b_gg | output_gate]

            input = x[:, cidx].unsqueeze(1)
            out, new_hidden = self.lstm(input, self.hidden)
            if cidx == x.size(1)-1:
                new_out, new_hidden = self.lstm(input, self.hidden)
                self.linear(new_out)

            self.hidden = new_hidden
            hs.append(self.hidden[0][0][0].detach().numpy())
            cs.append(self.hidden[1][0][0].detach().numpy())
            outs.append(out)

        x = torch.stack(outs)
        x = x.permute(1, 0, 3, 2)

        # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        # run through linear layer
        x = self.linear(x)

        # reshape to: (batch_size, seq_len, vocab_size)
        x = x.view(batch_size, seq_len, self.vocab_size)
        return x, cs, hs#, params

    def forward(self, x, x_lens):
        self.hidden = self.init_hidden()
        # self.apply_quantize()
        batch_size, seq_len = x.size()
        if self.use_embedding:
            x = self.embedding(x)
        x = x.unsqueeze(2).float()

        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, enforce_sorted=False, batch_first=True)
        x, self.hidden = self.lstm(x_packed, self.hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        # run through linear layer
        x = self.linear(x)

        # reshape to: (batch_size, seq_len, vocab_size)
        x = x.view(batch_size, seq_len, self.vocab_size)
        return x

    def compute_loss(self, preds, targets):
        """before we calculate the loss, mask out activations for padded elements."""
        targets = targets.view(-1, self.vocab_size)
        preds = preds.view(-1, self.vocab_size)

        # compute cross entropy loss which ignores all <PAD> tokens
        loss = self.criterion(preds, targets.argmax(dim=1))
        return loss



class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        hx, cx = hidden
        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


class LayerNormLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, vocab_size=None, num_layers=1, bias=True, bidirectional=False, linear_normal=False):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.target_pad_idx = 3
        self.vocab_size = vocab_size
        self.linear_normal = linear_normal
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

        if self.linear_normal:
            torch.nn.init.xavier_normal_(self.linear._parameters['weight'], gain=1.0)
        self.hiddencell = LayerNormLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)


    def init_hidden(self):
        h_0 = torch.zeros((self.batch_size, self.hidden_size))
        c_0 = torch.zeros((self.batch_size, self.hidden_size))
        h_0 = Variable(h_0)
        c_0 = Variable(c_0)
        return (h_0, c_0)

    def compute_loss(self, preds, targets):
        """before we calculate the loss, mask out activations for padded elements."""
        targets = targets.view(-1, self.vocab_size)
        preds = preds.view(-1, self.vocab_size)

        # compute cross entropy loss which ignores all <PAD> tokens
        loss = self.criterion(preds, targets.argmax(dim=1))
        return loss

    def forward(self, input):
        h0, c0 = self.init_hidden()
        if len(input.shape) == 2:
            input = input.unsqueeze(-1)

        input = input.transpose(0, 1)
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only

        ht = [None,] * seq_len
        ct = [None,] * seq_len

        for t, x in enumerate(input):
            if t == 0:
                ht[t], ct[t] = self.hiddencell(x, (h0, c0))
            else:
                ht[t], ct[t] = self.hiddencell(x, (ht[t-1], ct[t-1]))

        y = torch.stack(ht, dim=0)

        # run through linear layer
        y = self.linear(y.transpose(0, 1))

        # reshape to: (batch_size, seq_len, vocab_size)

        y = y.view(batch_size, seq_len, self.vocab_size)
        return y

    def forward_save(self, input):
        h0, c0 = self.init_hidden()
        if len(input.shape) == 2:
            input = input.unsqueeze(-1)
        input = input.transpose(0, 1)
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only

        ht = [None,] * seq_len
        ct = [None,] * seq_len

        for t, x in enumerate(input):
            if t == 0:
                ht[t], ct[t] = self.hiddencell(x, (h0, c0))
            else:
                ht[t], ct[t] = self.hiddencell(x, (ht[t-1], ct[t-1]))
        hs = torch.stack(ht, dim=0).detach().numpy()
        cs = torch.stack(ct, dim=0).detach().numpy()
        y = torch.stack(ht, dim=0)

        # run through linear layer
        y = self.linear(y.transpose(0, 1))

        # reshape to: (batch_size, seq_len, vocab_size)

        y = y.view(batch_size, seq_len, self.vocab_size)
        return y, cs, hs
