from .data import load
from .tree import DecisionTreeMethod
from .gmm import GaussianMixtureMethod

def main():
    training_data, training_z = load('training', thin=100)
    validation_data, validation_z = load('validation')
    nbin = 4
    T = DecisionTreeMethod(nbin,
                 training_data, training_z, 
                 validation_data, validation_z)
    z_edges, score = T.optimize(method='Nelder-Mead', tol=0.05)

    z_edges = ', '.join(f'{z:.2f}' for z in z_edges)
    print(f"nbin: {nbin} score: {score}")
    print(f"z: {z_edges}")


def main2():
    validation_data, validation_z = load('validation')
    nbin = 4
    G = GaussianMixtureMethod(nbin, validation_data, validation_z)
    score = G.run()
    print(f'GMM score: {score}')

if __name__ == '__main__':
    main()
