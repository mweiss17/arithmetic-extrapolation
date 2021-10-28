import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.autograd import Variable
from torch.nn import Parameter

class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size: int, learnable: bool = True, epsilon: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = torch.empty(1, input_size).fill_(0)
        self.beta = torch.empty(1, input_size).fill_(0)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - torch.mean(x, 1).unsqueeze(1)) / torch.sqrt(torch.var(x, 1).unsqueeze(1) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, vocab_size, device, bias=True, ln_preact=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.device = device
        self.target_pad_idx = 3
        self.bias = bias
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # if ln_preact:
        #     self.ln_i2h = LayerNorm(4 * hidden_size, learnable=True)
        #     self.ln_h2h = LayerNorm(4 * hidden_size, learnable=True)
        # self.ln_preact = ln_preact
        # self.ln_cell = LayerNorm(hidden_size, learnable=True)

        # [input_gate | forget_gate | b_gg | output_gate]
        # for names in self.lstm._all_weights:
        #     for name in filter(lambda n: "bias" in n, names):
        #         bias = getattr(self.lstm, name)
        #         n = bias.size(0)
        #         l = n // 4
        #         # set input gate bias
        #         bias.data[0:l].fill_(-.5)
        #         # set forget gate bias
        #         bias.data[l:l*2].fill_(1.)
        #         # set output gate bias
        #         bias.data[l*3:l*4].fill_(-1.)

        self.linear = nn.Linear(hidden_size, self.vocab_size)
        # self.linear._parameters['bias'] = Variable(torch.tensor([.48, .48, .04, 0]), requires_grad=True)
        torch.nn.init.xavier_normal_(self.linear._parameters['weight'], gain=1.0)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=1,
            padding_idx=0
        )

    def init_hidden(self):
        h_0 = torch.zeros((1, self.batch_size, self.hidden_size))
        c_0 = torch.zeros((1, self.batch_size, self.hidden_size))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        h_0 = Variable(h_0)
        c_0 = Variable(c_0)
        return (h_0, c_0)

    def apply_quantize(self):
        closest = lambda x: int(ceil(log(x) / log(2)))
        torch.clip(self.linear.weight, )
        import pdb; pdb.set_trace()
        print("asdf")

    def forward_save_hiddens(self, x):
        self.hidden = self.init_hidden()
        batch_size, seq_len = x.size()

        ### If we don't want embedding, do this
        x = x.unsqueeze(2).float()

        outs = []
        hiddens = []
        cells = []
        for cidx in range(x.size(1)):
            out, self.hidden = self.lstm(x[:, cidx].unsqueeze(1), self.hidden)
            hiddens.append(self.hidden[0].detach().numpy())
            cells.append(self.hidden[1].detach().numpy())
            outs.append(out)

        hiddens = np.array(hiddens).squeeze(1).squeeze(1)
        cells = np.array(cells).squeeze(1).squeeze(1)
        x = torch.stack(outs)
        x = x.permute(1, 0, 3, 2)

        # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        # run through linear layer
        x = self.linear(x)

        # reshape to: (batch_size, seq_len, vocab_size)
        x = x.view(batch_size, seq_len, self.vocab_size)
        return x, hiddens, cells

    def get_params(self):
        l = self.hidden_size
        linear_weights = self.linear.weight.view(-1).detach().numpy()
        linear_bias = self.linear.bias.view(-1).detach().numpy()

        weight_ih_l0 = self.lstm._parameters['weight_ih_l0'].detach().numpy() # (4 * hidden_size, input_size)
        weight_hh_l0 = self.lstm._parameters['weight_hh_l0'].detach().numpy() # (4 * hidden_size, hidden_size)

        input_gate_ih = weight_ih_l0[0:l].squeeze(0)
        input_gate_hh = weight_hh_l0[0:l].squeeze(0)
        forget_gate_ih = weight_ih_l0[l:l * 2].squeeze(0)
        forget_gate_hh = weight_hh_l0[l:l * 2].squeeze(0)
        gg_ih = weight_ih_l0[l * 2:l * 3].squeeze(0)
        gg_hh = weight_hh_l0[l * 2:l * 3].squeeze(0)
        output_gate_hh = weight_hh_l0[l * 3:l * 4].squeeze(0)
        output_gate_ih = weight_ih_l0[l * 3:l * 4].squeeze(0)

        bias_ih_l0 = self.lstm._parameters['bias_ih_l0'].detach().numpy()
        bias_hh_l0 = self.lstm._parameters['bias_hh_l0'].detach().numpy()

        bias_input_gate_ih = bias_ih_l0[0:l]
        bias_input_gate_hh = bias_hh_l0[0:l]
        bias_forget_gate_ih = bias_ih_l0[l:l*2]
        bias_forget_gate_hh = bias_hh_l0[l:l*2]
        bias_gg_ih = bias_ih_l0[l*2:l*3]
        bias_gg_hh = bias_hh_l0[l*2:l*3]
        bias_output_gate_ih = bias_ih_l0[l*3:l*4]
        bias_output_gate_hh = bias_hh_l0[l*3:l*4]

        params = {
                    "input_gate_ih": input_gate_ih,
                    "input_gate_hh": input_gate_hh,
                    "forget_gate_ih": forget_gate_ih,
                    "forget_gate_hh": forget_gate_hh,
                    "output_gate_ih": output_gate_ih,
                    "output_gate_hh": output_gate_hh,
                    "gg_ih": gg_ih,
                    "gg_hh": gg_hh,
                    "bias_input_gate_ih": bias_input_gate_ih,
                    "bias_input_gate_hh": bias_input_gate_hh,
                    "bias_forget_gate_ih": bias_forget_gate_ih,
                    "bias_forget_gate_hh": bias_forget_gate_hh,
                    "bias_gg_ih": bias_gg_ih,
                    "bias_gg_hh": bias_gg_hh,
                    "bias_output_gate_ih": bias_output_gate_ih,
                    "bias_output_gate_hh": bias_output_gate_hh,
                    "linear_weights": linear_weights,
                    "linear_bias": linear_bias,
        }
        return params

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

        ### If we want embedding, then do this
        # x = self.embedding(x)
        ### If we don't want embedding, do this
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


    def pad_collate(batch):
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
        for idx, y_len in enumerate(y_lens):
            target_pad_idx = 0
            yy_pad[idx, y_len:max(y_lens), target_pad_idx] = 1
        return xx_pad, yy_pad, x_lens, y_lens
