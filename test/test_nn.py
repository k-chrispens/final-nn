import numpy as np
import pytest
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs


def test_single_forward():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mse",
    )
    X = np.array([[1, 2]]).T  # need to transpose to match the shape of W
    b_curr = nn._param_dict["b1"]
    W_curr = nn._param_dict["W1"]
    activation = nn.arch[0]["activation"]
    activations, linear_transform = nn._single_forward(W_curr, b_curr, X, activation)
    assert activations.shape == (2, 1)
    assert linear_transform.shape == (2, 1)
    assert np.all(nn._relu(linear_transform) == activations)


def test_forward():
    nn = NeuralNetwork(
        nn_arch=[
            {"input_dim": 2, "output_dim": 3, "activation": "relu"},
            {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"},
        ],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mse",
    )
    X = np.array([[1, 2]])
    activations, cache = nn.forward(X)
    assert activations.shape == (1, 1)
    assert cache["A1"].shape == (3, 1)
    assert cache["Z1"].shape == (3, 1)
    assert cache["A2"].shape == (1, 1)
    assert cache["Z2"].shape == (1, 1)


def test_single_backprop():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mse",
    )
    X = np.array([[1, 2]]).T  # need to transpose to match the shape of W
    Y = np.array([[1, 0]]).T
    b_curr = nn._param_dict["b1"]
    W_curr = nn._param_dict["W1"]
    activation = nn.arch[0]["activation"]
    activations, linear_transform = nn._single_forward(W_curr, b_curr, X, activation)
    dA_prev, dW_curr, db_curr = nn._single_backprop(
        W_curr, b_curr, Y, activations, linear_transform, activation
    )
    assert dA_prev.shape == (2, 1)
    assert dW_curr.shape == (2, 2)
    assert db_curr.shape == (2, 1)


def test_predict():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mse",
    )
    nn._param_dict["W1"] = np.array([[1, 0], [0, 1]])
    nn._param_dict["b1"] = np.array([[1], [1]])
    X = np.array([[1, 2]])
    Y = nn.predict(X)
    Y = nn._relu(Y)
    assert Y.shape == (1, 2)
    assert np.all(
        Y == np.array([[2, 3]])
    )  # manually calculated output from X and W1, b1


def test_binary_cross_entropy():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="bce",
    )
    y_hat = np.array([0.9, 0.2, 0.1])
    y = np.array([1, 0, 0])
    loss = nn._binary_cross_entropy(y, y_hat)
    expected_loss = np.mean([-np.log(0.9), -np.log(0.8), -np.log(0.9)])
    assert np.allclose(loss, expected_loss, atol=1e-5)


def test_binary_cross_entropy_backprop():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="bce",
    )
    y_hat = np.array([0.9, 0.2, 0.1])
    y = np.array([1, 0, 0])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    expected_dA = np.array([-1 / 0.9, 1 / 0.8, 1 / 0.9])
    assert np.allclose(dA, expected_dA, atol=1e-5)


def test_mean_squared_error():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mse",
    )
    y_hat = np.array([0.9, 0.2, 0.1])
    y = np.array([1, 0, 0])
    loss = nn._mean_squared_error(y, y_hat)
    expected_loss = np.mean([(1 - 0.9) ** 2, (0 - 0.2) ** 2, (0 - 0.1) ** 2])
    assert np.allclose(loss, expected_loss, atol=1e-5)


def test_mean_squared_error_backprop():
    nn = NeuralNetwork(
        nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="mse",
    )
    y_hat = np.array([0.9, 0.2, 0.1])
    y = np.array([1, 0, 0])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    expected_dA = np.array([2 * (0.9 - 1), 2 * (0.2 - 0), 2 * (0.1 - 0)])
    assert np.allclose(dA, expected_dA, atol=1e-5)


def test_sample_seqs():
    seqs = ["ATCCAA", "ATCGAA", "ATCGTA", "ATCGAC", "ATCGAT", "ATCGAG", "ATCCAA"]
    labels = [0, 1, 1, 1, 1, 1, 0]
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)
    assert len(sampled_seqs) == len(sampled_labels)
    assert len([label for label in sampled_labels if label == 0]) == len([label for label in sampled_labels if label == 1])


def test_one_hot_encode_seqs():
    seqs = ["ATC", "TAG", "ATA"]
    encodings = one_hot_encode_seqs(seqs)
    assert encodings.shape == (3, 12)
    assert np.all(encodings[0] == np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))
    assert np.all(encodings[1] == np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]))
    assert np.all(encodings[2] == np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]))
