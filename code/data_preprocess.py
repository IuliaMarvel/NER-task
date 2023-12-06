import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def build_index_map(sentences, pad=0):
  vocab = set(word for sentence in sentences for word in sentence)
  word2idx = {word: idx + pad for idx, word in enumerate(vocab)}
  return word2idx


def convert_to_idxs(sequence, idx_map):
  unknown_idx = len(idx_map) + 1
  sequence = [idx_map[el] if el in idx_map.keys() else unknown_idx for el in sequence]
  return sequence


def pad_sequence(sequence, max_len):
  #  zero is index of padding element
  sequences_padded = sequence[:max_len] + [0] * (max_len - len(sequence[:max_len]))
  return torch.tensor(sequences_padded)


class CustomNERDataset(Dataset):
  def __init__(self, sentences, labels, max_len, label2idx=None, word2idx=None):
    self.sentences = sentences
    self.labels = labels
    self.max_len = min(max_len, max(len(line) for line in self.sentences))
    if label2idx is None:
      self.label2idx = build_index_map(self.labels, pad=1)
    else:
      self.label2idx = label2idx
    if word2idx is None:
      self.word2idx = build_index_map(self.sentences, pad=1)
    else:
      self.word2idx = word2idx

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, idx):
    sentence = pad_sequence(convert_to_idxs(self.sentences[idx], self.word2idx), self.max_len)
    labels_for_sentence = pad_sequence(convert_to_idxs(self.labels[idx], self.label2idx), self.max_len)

    return sentence, labels_for_sentence



def get_train_test_loaders_info(sentences, labels):
  x_train, x_test, y_train, y_test = train_test_split(sentences, labels, random_state=42)
  train_dataset = CustomNERDataset(x_train, y_train, max_len=20)
  word2idx, label2idx = train_dataset.word2idx, train_dataset.label2idx
  vocab_info = {}
  vocab_info['vocab_size'] = len(word2idx) + 2 # padding + unknown
  vocab_info['output_size'] = len(label2idx) + 2 # padding + unknown
  test_dataset = CustomNERDataset(x_test, y_test, max_len=20, label2idx=label2idx, word2idx=word2idx)
  train_loader = DataLoader(train_dataset, batch_size=1, num_workers=3)
  test_loader = DataLoader(test_dataset, batch_size=1, num_workers=3)
  return train_loader, test_loader, vocab_info


