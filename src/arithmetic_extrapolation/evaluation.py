import torch
import numpy as np

def get_acc(dataloader, model, use_ln, split):
    char_accs = []
    for xx_pad, targets, x_lens, _ in dataloader:
        if use_ln:
            preds = model(xx_pad.unsqueeze(2).float())
        else:
            preds = model(xx_pad, x_lens)
        loss = model.compute_loss(preds, targets)
        for i in range(preds.shape[0]):
            pred = preds[i, :x_lens[i]]
            target = targets[i, :x_lens[i]]
            char_matches = torch.sum(pred.argmax(1) == target.argmax(1), dtype=torch.float32)
            char_acc = (torch.div(char_matches, pred.shape[0]) * 100).detach().numpy().item()
            char_accs.append(char_acc)
    char_acc = np.mean(char_accs)
    word_acc = np.mean([a == 100. for a in char_accs]) * 100.
    result_dict = {f"wa-{split}": word_acc, f"ca-{split}": char_acc, f"loss-{split}": loss.detach().item()}
    return result_dict

def get_char_acc(pred, target):
    for i in range(pred.shape[0]):
        char_matches = torch.sum(pred.argmax(1) == target.argmax(1), dtype=torch.float32)
        char_acc = (torch.div(char_matches, pred.shape[0]) * 100).detach().numpy().item()
    result_dict = {f"ca": char_acc}
    return result_dict
#
# def convert_preds_to_words(preds, x_lens, ix_to_char):
#     words = []
#     for i in range(preds.size(0)):
#         word = ""
#         for c in preds[i]:
#             char = ix_to_char[c.argmax(0).item()]
#             word = word + char
#         words.append(word[:x_lens[i]])
#     return words
#
#
# def convert_targets_to_words(targets, ix_to_char):
#     words = []
#     for i in range(targets.size(0)):
#         word = ""
#         target = targets[i].argmax(1).detach().numpy()
#         letters = [ix_to_char[c.item()] for c in target]
#         for char in letters:
#             if char == "<PAD>":
#                 continue
#             word = word + char
#         words.append(word)
#     return words
#
#
# def preds_and_targets_to_words(preds, targets, x_lens, ix_to_char):
#     all_target_words = []
#     all_pred_words = []
#     for i in range(len(x_lens)):
#         words = convert_preds_to_words(preds[i], x_lens[i], ix_to_char)
#         all_pred_words.extend(words)
#         words = convert_targets_to_words(targets)
#         all_target_words.extend(words)
#     return all_pred_words, all_target_words
#
#
# def get_all_words(dataloader, model, ix_to_char):
#     all_target_words = []
#     all_pred_words = []
#     for xx_pad, targets, x_lens, _ in dataloader:
#         preds = model(xx_pad, x_lens, record_activations=False)
#         word = convert_preds_to_words(preds, x_lens, ix_to_char)
#         all_pred_words.append(word)
#         word = convert_targets_to_words(targets, ix_to_char)
#         all_target_words.append(word)
#     return all_pred_words, all_target_words
