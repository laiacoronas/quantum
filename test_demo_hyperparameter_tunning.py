import pandas as pd

from unique.ml.quantum.pennylane.model.tunning.hyperparameter_tunning import HyperparameterTunning
from unique.infrastructure.quantum.pennylane.embedding.angle_embedding import AngleEmbedding
from unique.infrastructure.quantum.pennylane.vendor.local_simulator import LocalSimulator


def get_sample_input():
    # T = (A + B) * C + D
    df = pd.DataFrame([
        [1, 6.5, "uno", 1, 8.5],  # T = (1 + 6.5) * 1 + 1 = 8.5
        [2, 4.5, "dos", 2, 15.0],  # T = (2 + 4.5) * 2 + 2 = 15.0
        [3, 6.5, "uno", 3, 12.5],  # T = (3 + 6.5) * 1 + 3 = 12.5
        [4, 2.5, "uno", 1, 7.5],  # T = (4 + 2.5) * 1 + 1 = 7.5
        [5, 2.5, "dos", 2, 17.0],  # T = (5 + 2.5) * 2 + 2 = 17.0
        [6, 6.5, "uno", 1, 8.5],  # T = (6 + 6.5) * 1 + 1 = 13.5
        [7, 4.5, "dos", 1, 24.0],  # T = (7 + 4.5) * 2 + 1 = 24.0
        [8, 6.5, "uno", 3, 17.5],  # T = (8 + 6.5) * 1 + 3 = 17.5
        [9, 2.5, "dos", 1, 24.0],  # T = (9 + 2.5) * 2 + 1 = 24.0
        [10, 2.5, "dos", 2, 27.0],  # T = (10 + 2.5) * 2 + 2 = 27.0
        [1, 6.5, "uno", 0, 7.5],  # T = (1 + 6.5) * 1 + 0 = 7.5
        [2, 4.5, "dos", 0, 13.0],  # T = (2 + 4.5) * 2 + 0 = 13.0
        [3, 6.5, "uno", 0, 9.5],  # T = (3 + 6.5) * 1 + 0 = 9.5
        [4, 1.0, "dos", 0, 12.0]],  # T = (4 + 1.0) * 2 + 2 = 12.0
        columns=["A", "B", "C", "D", "T"])
    return AngleEmbedding.from_whole_df(df, columns_target="T",
                                        to_quantum=True, verbose=False)


def test_hyperparams_local():
    # get features
    features = get_sample_input()
    # train
    ht = HyperparameterTunning()
    ht.train(features=features,
             vendor=LocalSimulator(backend_name="default.qubit", verbose=True),
             shots=1000,
             max_epochs=3,
             optimizer_stepsize=0.15,
             optimizer_beta1=0.8,
             optimizer_beta2=0.85,
             random_seed=42)
    # evaluate
    ht.evaluate(use_optimal_x=True,
                shots=1000,
                max_epochs=3,
                optimizer_stepsize=0.0005,
                optimizer_beta1=0.79,
                optimizer_beta2=0.999,
                random_seed=42)
    # make sure hyper params exist
    hyper_parameters = features.to_classic(ht.x)
    return hyper_parameters


hyper_parameters = test_hyperparams_local()
print(hyper_parameters)