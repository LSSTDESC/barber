from .data import load
from .tree import DecisionTreeMethod


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


if __name__ == '__main__':
    main()
