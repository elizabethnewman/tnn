import argparse
import math

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--M', type=str, default='dct', help='transformation matrix')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_train', type=int, default=1000, help='number of training points')
    parser.add_argument('--n_val', type=int, default=10000, help='number of validation points')
    parser.add_argument('--n_test', type=int, default=10000, help='number of test points')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='decay rate for scheduler')
    parser.add_argument('--step_size', type=float, default=100, help='step size for scheduler')
    parser.add_argument('--width', type=int, default=10, help='width of network')
    parser.add_argument('--auto_width', type=int, default=20, help='autoencoder additional width of network')
    parser.add_argument('--depth', type=int, default=4, help='depth of network')
    parser.add_argument('--h_step', type=float, default=0.1, help='number of steps in Hamiltonian')
    parser.add_argument('--alpha', type=float, default=0.0, help='regularization parameter')
    parser.add_argument('--add_width_hamiltonian', type=int, default=0, help='additional width of Hamiltonian network')
    parser.add_argument('--opening_layer', action='store_true', help='opening linear layer')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--bias', type=bool, default=True, help='bias for network')
    parser.add_argument('--matrix_match_tensor', action='store_true',
                        help='match number of matrix weights to tensor based on width')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    return parser


def matrix_match_tensor_single_layer(width, loss='t_cross_entropy'):
    W, H, n_classes = 28, 28, 10
    if loss == 'cross_entropy':
        # tensor: (W * H + H + n_classes * H) * width + n_classes
        # first layer: width x W x H + width x 1 x H
        # final layer (vectorize): n_classes x width * H + n_classes x 1

        # matrix: (W * H + 1 + n_classes) * width2 + n_classes
        # first layer: width2 * W * H + width2 x 1
        # final layer: n_classes x width2 + n_classes x 1

        # we return width2 such that the total number of parameters is similar and we give more to matrix
        return math.ceil((W * H + H + n_classes * H) * width / (W * H + 1 + n_classes))

    elif loss == 't_cross_entropy':
        # tensor: (W * H + H + n_classes * H) * width + n_classes * H
        # first layer: width x W x H + width x 1 x H
        # final layer: n_classes x width x H + n_classes x 1 x H

        # matrix: (W * H + 1 + n_classes) * width2 + n_classes
        # first layer: width2 * W * H + width2 x 1
        # final layer: n_classes x width2 + n_classes x 1

        return math.ceil(((W * H + H + n_classes * H) * width + n_classes * H) / (W * H + 1 + n_classes + n_classes))

    else:
        return 0


if __name__ == '__main__':
    for width in range(2, 11):
        print([matrix_match_tensor_single_layer(width, 't_cross_entropy'), matrix_match_tensor_single_layer(width, 'cross_entropy')])

