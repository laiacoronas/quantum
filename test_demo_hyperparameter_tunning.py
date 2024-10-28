import pandas as pd

from unique.ml.quantum.pennylane.model.tunning.hyperparameter_tunning import HyperparameterTunning
from unique.infrastructure.quantum.pennylane.embedding.angle_embedding import AngleEmbedding
from unique.infrastructure.quantum.pennylane.vendor.local_simulator import LocalSimulator


def get_sample_input():
    # T = (A + B) * C + D
    df = pd.read_csv("C:\\Users\\lclai\\Desktop\\LDIG\\DB\\training\\prova.csv")
    return AngleEmbedding.from_whole_df(df, columns_target="Eat",
                                        to_quantum=True, verbose=False)
    
    
#### jenfwoinbwowefnwioffnwfi



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