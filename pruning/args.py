import argparse
import models

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pruning FC Layers with Redundant Links')
    
    # Hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='Input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='Number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='Momentum (default: 0.9)'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=0.0005,
        metavar='M',
        help='Weight decay (default: 0.0005)'
    )

    # Architecture and training
    parser.add_argument(
        '--arch',
        type=str,
        default='RedTwoLayerFC',
        help='Model architecture: ' + ' | '.join(models.__dict__['__all__']) + ' | (default: RedTwoLayerFC)'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=500,
        metavar='H',
        help='Number of nodes in the FC layers of the network. Important: when the model of choice is a wide network, the number of hidden nodes is determined based on this argument and the --redundancy argument. This allows for determining how many hidden nodes the wide network should have in order to have the same number of parameters as a network with our structure. (default: 500)'
    )
    parser.add_argument(
        '--use-relu',
        action='store_true',
        default=False,
        help='If true, additional ReLU activation will be included (as descirbed in our paper) when adding redundancy'
    )
    parser.add_argument(
        '--sparsity',
        type=float,
        default=0.5,
        metavar='S',
        help='The ratio of weights to remove in each fully connected layer. A sparsity of 0 means no weights are pruned, and a sparsity of 1 means all weights are pruned (default: 0.5)'
    )
    parser.add_argument(
        '--redundancy',
        type=int,
        default=5,
        metavar='R',
        help='Number of units of redundancy; must be greater than 0. A redundancy of 1 provides one link per connection (same number of parameters as baseline), a redundancy of 2 provides one extra link per connection (twice the number of parameters as baseline, not including bias), etc. (default: 5)'
    )
    parser.add_argument(
        '--bias',
        action='store_true',
        default=None,
        help='Boolean flag to indicate the inclusion of bias terms in the neural network (default: None)'
    )
    parser.add_argument(
        '--freeze-weights',
        action='store_true',
        default=False,
        help='Boolean flag to indicate whether weights should be frozen. Used when sparsifying'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='MNIST',
        help='Dataset to train the model with. Can be CIFAR10 or MNIST (default: MNIST)'
    )
    
    # Save/Load
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=False,
        help='Save the trained model'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        default=False,
        help='Save the results and arguments of this run'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        metavar='N',
        help='The number of batches to wait before logging training status'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory of dataset'
    )
    parser.add_argument(
        '--load-weights',
        type=str,
        default=None,
        help='Provide a path to load network weights from; only used if --arch is "RedFCLeNet5" or "WideFCLeNet5'
    )
   
    # Device settings
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='Disables CUDA training'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        metavar='G',
        help='Override the default choice for a CUDA-enabled GPU by specifying the GPU\'s integer index (i.e. "0" for "cuda:0")'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='Random seed (default: 1)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        metavar='M',
        help='Number of workers'
    )

    return parser.parse_args()


def run_args():
    global args
    args = parse_arguments()


run_args()
