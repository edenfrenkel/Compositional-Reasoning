import torch
from torch.utils.data import TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score


class CompositionalDataset:
    def __init__(self, class_count=10, seq_length=10, overlap=2, noise=None):
        self.class_count = class_count
        self.seq_length = seq_length
        self.overlap = overlap
        self.comp_count, self.comps = self._create_compositions()
        self.noise = noise

    def _create_compositions(self):
        length = self.class_count ** 2
        before = np.reshape(np.arange(length, dtype=np.int), (self.class_count, self.class_count))

        during = []
        for i in range(self.class_count):
            during.append(np.arange(length, length + self.class_count - i, dtype=np.int))
            length += self.class_count - i

        return length, [before, during]

    def _get_labels(self, seq):
        # Sequence shape: seq_length x class_count
        labels = set()
        prev_classes = set()

        for el in seq:
            non_zero = np.nonzero(el.numpy())[0]
            # Set BEFORE labels
            for cls in prev_classes:
                for i in non_zero:
                    labels.add(self.comps[0][cls, i])
            # Set DURING labels
            for i in range(len(non_zero)):
                for j in range(i, len(non_zero)):
                    labels.add(self.comps[1][non_zero[i]][non_zero[j] - non_zero[i]])
            # Add current to previous
            prev_classes.update(non_zero)

        label_vec = torch.zeros(self.comp_count, dtype=torch.float)
        label_vec[list(labels)] = 1

        return label_vec

    def _generate_seq(self):
        classes = np.random.choice(self.class_count, size=(self.seq_length, self.overlap))
        seq = torch.zeros((self.seq_length, self.class_count), dtype=torch.float)
        for i in range(self.seq_length):
            seq[i, classes[i]] = 1
        label = self._get_labels(seq)

        if self.noise is not None:
            noise = torch.rand(seq.size()) * self.noise
            noise[seq == 1] *= -1
            seq += noise

        return seq, label

    def generate_dataset(self, count=8192):
        seqs = torch.empty((count, self.seq_length, self.class_count), dtype=torch.float)
        labels = torch.empty((count, self.comp_count), dtype=torch.float)

        for i in tqdm(range(count)):
            seqs[i], labels[i] = self._generate_seq()

        return TensorDataset(seqs, labels)
