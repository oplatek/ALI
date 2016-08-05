#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg') 
import argparse

import theano
import theano.tensor as tt
import numpy as np
from blocks.serialization import load
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid

from ali import streams


def main(main_loop, data_stream, nrows, ncols, save_path=None, sigma=1.0, mu=0.0):
    ali, = main_loop.model.top_bricks
    input_shape = ali.encoder.get_dim('output')

    z = tt.tensor3('state', dtype='float32')
    x = ali.sample(z)
    f = theano.function([z], x)

    examples, = next(data_stream.get_epoch_iterator())
    orig_img = examples[0]
    noises = [sigma * np.random.randn(*input_shape) + mu for _ in range(nrows*ncols)]

    samples = [f(orig_img + noise) for noise in noises]

    figure = pyplot.figure()
    grid = ImageGrid(figure, 111, (nrows, ncols), axes_pad=0.1)

    for sample, axis in zip(samples, grid):
        axis.imshow(sample.transpose(1, 2, 0).squeeze(),
                    cmap=cm.Greys_r, interpolation='nearest')
        axis.set_yticklabels(['' for _ in range(sample.shape[1])])
        axis.set_xticklabels(['' for _ in range(sample.shape[2])])
        axis.axis('off')

    if save_path is None:
        pyplot.show()
    else:
        pyplot.savefig(save_path, transparent=True, bbox_inches='tight')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot samples.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    parser.add_argument("--nrows", type=int, default=10,
                        help="number of rows of samples to display.")
    parser.add_argument("--ncols", type=int, default=10,
                        help="number of columns of samples to display.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the generated samples.")
    args = parser.parse_args()

    num_examples = args.nrows * args.ncols
    rng = np.random.RandomState()
    _1, _2, data_stream = streams.create_cifar10_data_streams(num_examples,
                                                              num_examples,
                                                              rng=rng)

    with open(args.main_loop_path, 'rb') as src:
        main(load(src), data_stream, args.nrows, args.ncols, args.save_path)
