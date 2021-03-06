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


def main(main_loop, data_stream, num_pairs, num_steps, save_path):
    ali, = main_loop.model.top_bricks
    x = tensor.tensor4('features')
    z = tensor.tensor4('z')
    encode = theano.function([x], ali.encoder.apply(x))
    decode = theano.function([z], ali.decoder.apply(z))

    it = data_stream.get_epoch_iterator()
    (from_x,), (to_x,) = next(it), next(it)
    from_z, to_z = encode(from_x), encode(to_x)

    from_to_tensor = to_z - from_z
    between_x_list = [from_x]
    for alpha in numpy.linspace(0, 1, num_steps + 1):
        between_z = from_z + alpha * from_to_tensor
        between_x_list.append(decode(between_z))
    between_x_list.append(to_x)

    figure = pyplot.figure()
    grid = ImageGrid(figure, 111, (num_pairs, num_steps + 3), axes_pad=0.1)
    images = numpy.empty(
        (num_pairs * (num_steps + 3),) + between_x_list[0].shape[1:],
        dtype=between_x_list[0].dtype)
    for i, between_x in enumerate(between_x_list):
        images[i::num_steps + 3] = between_x

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
    parser = argparse.ArgumentParser(description="Plot interpolations.")
    parser.add_argument("which_dataset", type=str,
                        choices=tuple(stream_functions.keys()),
                        help="which dataset to compute interpolations on.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    parser.add_argument("--num-pairs", type=int, default=10,
                        help="number of pairs of samples to interpolate.")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="number of interpolation steps.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the interpolations.")
    args = parser.parse_args()

    with open(args.main_loop_path, 'rb') as src:
        main_loop = load(src)
    num_pairs = args.num_pairs
    rng = numpy.random.RandomState()
    _1, _2, data_stream = stream_functions[args.which_dataset](num_pairs,
                                                               num_pairs,
                                                               rng=rng)
    main(main_loop, data_stream, num_pairs, args.num_steps, args.save_path)
