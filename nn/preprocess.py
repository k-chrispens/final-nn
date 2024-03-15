# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance.
    Consider this a sampling scheme with replacement.

    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label]
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]

    if len(pos_seqs) > len(neg_seqs):
        minority_seqs, minority_labels = neg_seqs, [False] * len(neg_seqs)
        majority_seqs, majority_labels = pos_seqs, [True] * len(pos_seqs)
    else:
        minority_seqs, minority_labels = pos_seqs, [True] * len(pos_seqs)
        majority_seqs, majority_labels = neg_seqs, [False] * len(neg_seqs)

    # Number of samples needed to balance the classes
    num_samples_needed = len(majority_seqs) - len(minority_seqs)

    # Upsample the minority class with replacement # NOTE: I asked Tony, and he mentioned upsampling is what we want
    sampled_minority_indices = np.random.choice(
        len(minority_seqs), size=num_samples_needed, replace=True
    )
    sampled_minority_seqs = [minority_seqs[i] for i in sampled_minority_indices] + minority_seqs
    sampled_minority_labels = [minority_labels[i] for i in sampled_minority_indices] + minority_labels

    # Combine the sampled minority class with the majority class
    sampled_seqs = majority_seqs + sampled_minority_seqs
    sampled_labels = majority_labels + sampled_minority_labels

    # Shuffle the combined sequences and labels
    combined = list(zip(sampled_seqs, sampled_labels))
    np.random.shuffle(combined)
    sampled_seqs, sampled_labels = zip(*combined)

    return list(sampled_seqs), list(sampled_labels)


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    encodings = np.zeros(
        (
            len(seq_arr),
            4 * len(seq_arr[0]),
        )
    )  # Build flattened array from the beginning
    for i, seq in enumerate(seq_arr):
        if not set(seq).issubset({"A", "T", "C", "G"}):
            raise ValueError(f"Invalid sequence: {seq}")
        for j, base in enumerate(seq):
            if base not in "ATCG":
                raise ValueError(f"Invalid base: {base}")
            if base == "A":
                encodings[i, 4 * j] = 1
            elif base == "T":
                encodings[i, 4 * j + 1] = 1
            elif base == "C":
                encodings[i, 4 * j + 2] = 1
            elif base == "G":
                encodings[i, 4 * j + 3] = 1
    return encodings
