import argparse


def ParseInput():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='enable training')

    """ ---------- Simulation Parameters ---------- """
    parser.add_argument('--sigma', type=float, default=1.0, dest='sigma', help='non-linearity')
    parser.add_argument('--K', type=int, default=5, dest='K', help='neighbours')
    parser.add_argument('--minN', type=int, default=5, dest='minN', help='minimumN')
    parser.add_argument('--numN', type=int, default=20, dest='numN', help='numberN')
    parser.add_argument('--divN', type=int, default=5, dest='divN', help='divisionN')
    parser.add_argument('--maxSeed', type=int, default=25, dest='maxSeed', help='numberSeed')

    args = parser.parse_args()

    """ ---------- Algorithm Hyperparameters -------"""

    args.num_actions = 2
    args.num_states = 10
    args.J = 10 ** 2                 # Number of iterations for training the neural network based policy
    args.L = 10 ** 2
    args.run_eval = 10 ** 2          # Number of iterations for evaluating a policy
    args.gamma = 0.9                 # Discount factor

    """ --------- Reward Parameters --------- """
    args.alpha_r = 1
    args.beta_r = 0.5
    args.lambda_r = 0.5

    """----------- Learning Parameters --------- """
    args.alpha = 10**-3
    args.eta = 10**-3
    args.hidden_size = 32

    return args
