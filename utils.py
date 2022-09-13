import random
from enum import Enum
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


class Task(Enum):
    COOKING = 0
    WALKING = 1
    DRIVING = 2


class Gender(Enum):
    MAN = 0
    WOMAN = 1


MAN_BIAS = {
    Task.COOKING: 0.34,
    Task.WALKING: 0.5,
    Task.DRIVING: 0.6,
}

SAMPLES_PER_COMB_BASE = 50000

COMBINATIONS = [(t, g) for t in Task for g in Gender]


CORRECT_TASK_PREDICTION = 0.95
CORRECT_GENDER_PREDICTION = 0.8
MAN_PREDICTION_BIAS = {
    Task.COOKING: 0.2,
    Task.WALKING: 0.5,
    Task.DRIVING: 0.7,
}


def generate_biased_samples():
    samples = []
    for t in Task:
        for g in Gender:
            num_samples = SAMPLES_PER_COMB_BASE
            b = MAN_BIAS[t]
            num_samples *= b if g == Gender.MAN else 1 - b
            samples.extend([(t, g) for _ in range(int(num_samples))])
    random.shuffle(samples)
    return samples


# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))


def predict_with_randomness(n, p=None, w=1):
    """Gets a probability vector of length n with the highest value on index p and an additional weighting by w"""
    randoms = np.random.random(n) * w
    if p is not None:
        randoms[p] += 2
    return softmax(randoms)


def predict(real_task, real_gender):
    """Gets biased predictions"""
    task_prediction = predict_with_randomness(
        len(Task),
        real_task.value if np.random.random() < CORRECT_TASK_PREDICTION else None,
    )
    gender_prediction = predict_with_randomness(
        len(Gender),
        real_gender.value if np.random.random() < CORRECT_GENDER_PREDICTION else None,
    )

    joint_prediction = task_prediction[:, None] * gender_prediction
    man_bias_vec = np.array(list(MAN_PREDICTION_BIAS.values()))
    joint_prediction[:, Gender.MAN.value] += man_bias_vec - 0.5
    joint_prediction[:, Gender.WOMAN.value] -= man_bias_vec - 0.5
    return joint_prediction / joint_prediction.sum()


def get_count_dict(data):
    """Counts the number of predictions for every combination and puts it in a nested dict"""
    count_tuples = Counter(data)
    count_dict = defaultdict(defaultdict)
    for k, v in count_tuples.items():
        current = count_dict
        for part in k[:-1]:
            current = current[part]
        current[k[-1]] = v
    return count_dict


def plot_ratio_comparisons(dicts_1, dicts_2, ratio_key_a, ratio_key_b, margin=None):
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    f = plt.gca()
    f.set_xlim([0, 1])
    f.set_ylim([0, 1])
    f.plot([0, 1], [0, 1], "b")
    f.set_xlabel("True ratio")
    f.set_ylabel("Predicted ratio")
    if margin is not None:
        f.plot([0, 1], [-margin, 1 - margin], "b--")
        f.plot([0, 1], [margin, 1 + margin], "b--")
    for k in dicts_1:
        d1 = dicts_1[k]
        d2 = dicts_2[k]
        f.scatter(
            d1[ratio_key_a] / (d1[ratio_key_a] + d1[ratio_key_b]),
            d2[ratio_key_a] / (d2[ratio_key_a] + d2[ratio_key_b]),
        )


def get_predictions_from_probs(predictions_probs):
    predictions_idx = np.argmax(
        predictions_probs.reshape(predictions_probs.shape[0], -1), 1
    )
    predictions = [COMBINATIONS[p] for p in predictions_idx]
    return predictions


def accuracy(truths, predictions):
    assert len(truths) == len(
        predictions
    ), "Ground truths and predictions must have same length"
    s = 0
    for t, p in zip(truths, predictions):
        if t == p:
            s += 1
    return s / len(truths)
