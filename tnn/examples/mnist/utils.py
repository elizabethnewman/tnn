
import math

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

