from .data import load
from .tree import DecisionTreeMethod
from .gmm import GaussianMixtureMethod

def main():
    training_data, training_z = load('training', thin=100)
    validation_data, validation_z = load('validation')
    nbin = 4

    # use a purity requirement for main bins
    purity_test = True

    # number of steps in the tree.
    # try to avoid over-fitting
    max_depth = 20
    
    # starting value for purity requirement for main bins.
    # Indicates the classifier should be at least 90% sure of the
    # classification
    purity_guess = [0.5]

    # Initialize the classifier
    T = DecisionTreeMethod(nbin,
                           training_data, training_z, 
                           validation_data, validation_z,
                           purity_test=purity_test,
                           max_depth=max_depth
    )

    # Run the optimizer
    z_edges, rej_param, score = T.optimize(extra_starts=purity_guess,
                                           method='Nelder-Mead',
                                           tol=0.01)

    z_edges = ', '.join(f'{z:.2f}' for z in z_edges)
    print(f"nbin: {nbin} score: {score}")
    print(f"z: {z_edges}")
    print(f"rejection threshold: {rej_param}")


def main2():
    validation_data, validation_z = load('validation')
    nbin = 4
    G = GaussianMixtureMethod(nbin, validation_data, validation_z)
    score = G.run()
    print(f'GMM score: {score}')

if __name__ == '__main__':
    main()
