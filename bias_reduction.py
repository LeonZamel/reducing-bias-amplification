import numpy as np


def lagrangian_subgradient(
    original_predictions, constraints_A, constraints_b, iterations, learning_rate
):
    """The core algorithm used to calibrate outputs and reduce bias"""
    # Assumes that sum of all y values is one vector
    lambdas = np.zeros_like(constraints_b)
    predictions = None

    for _ in range(iterations):
        predictions = original_predictions.copy() - lambdas * (
            constraints_A.sum(1) - constraints_b
        )

        # Get the top one prediction and convert to one hot
        y = np.eye(constraints_A.shape[1])[np.argmax(predictions, -1)]

        # Determine sum of violated constraints
        violations = (constraints_A @ y.T - constraints_b).sum(axis=1)
        if (violations > 0).sum() == 0:
            # No constraint violated, end early
            break

        # Update lambda weights
        lambdas = np.maximum(0, lambdas + learning_rate * violations)

    return predictions
