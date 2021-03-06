#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg') 
import argparse

import numpy
import theano
from blocks.serialization import load
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid
from theano import tensor

from ali import streams


def main(main_loop, data_stream, nrows, ncols, save_path, sigma):
    ali, = main_loop.model.top_bricks
    x = tensor.tensor4('features')
    orig_examples, = next(data_stream.get_epoch_iterator())
    print('orig_examples', orig_examples.shape)

    print(orig_examples[0])

    examples = numpy.array([orig_examples[0] + sigma * numpy.random.randn(*orig_examples[0].shape) for _ in orig_examples]).astype('float32')
    print('examples', examples.shape)

    reconstructions = theano.function([x], ali.reconstruct(x))(examples)
    print('rec', reconstructions.shape)
    print(reconstructions[0])

    figure = pyplot.figure()
    grid = ImageGrid(figure, 111, (nrows, ncols), axes_pad=0.1)
    images = numpy.concatenate([numpy.array([orig_examples[0]]), reconstructions[1:]], axis=0)
    print('images', images.shape)
    #images = numpy.empty(
    #    (2 * nrows * ncols,) + examples.shape[1:], dtype=examples.dtype)
    #images[::2] = examples
    #images[1::2] = reconstructions

    for image, axis in zip(images, grid):
        axis.imshow(image.transpose(1, 2, 0).squeeze(),
                    cmap=cm.Greys_r, interpolation='nearest')
        axis.set_yticklabels(['' for _ in range(image.shape[1])])
        axis.set_xticklabels(['' for _ in range(image.shape[2])])
        axis.axis('off')

    if save_path is None:
        pyplot.show()
    else:
        pyplot.savefig(save_path, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    stream_functions = {
        'cifar10': streams.create_cifar10_data_streams,
        'svhn': streams.create_svhn_data_streams,
        'celeba': streams.create_celeba_data_streams,
        'tiny_imagenet': streams.create_tiny_imagenet_data_streams}
    parser = argparse.ArgumentParser(description="Plot reconstructions.")
    parser.add_argument("which_dataset", type=str,
                        choices=tuple(stream_functions.keys()),
                        help="which dataset to compute reconstructions on.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    parser.add_argument("--nrows", type=int, default=10,
                        help="number of rows of samples to display.")
    parser.add_argument("--ncols", type=int, default=10,
                        help="number of columns of samples to display.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the reconstructions.")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="sqrt(variance) of the noise added")
    args = parser.parse_args()

    with open(args.main_loop_path, 'rb') as src:
        main_loop = load(src)
    num_examples = args.nrows * args.ncols
    rng = numpy.random.RandomState()
    _1, _2, data_stream = stream_functions[args.which_dataset](num_examples,
                                                               num_examples,
                                                               rng=rng)
    main(main_loop, data_stream, args.nrows, args.ncols, args.save_path, args.sigma)
