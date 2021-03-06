#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg') 
import argparse
import itertools

import numpy
import theano
from blocks.serialization import load
from matplotlib import pyplot, rc
from theano import tensor

from ali.datasets import GaussianMixture

rc('font', **{'family': 'serif', 'serif': 'Computer Modern Roman'})
rc('text', usetex=True)

MEANS = [numpy.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                           range(-4, 5, 2))]
VARIANCES = [0.05 ** 2 * numpy.eye(len(mean)) for mean in MEANS]
PRIORS = None


def main(ali_main_loop, gan_main_loop, save_path=None):
    ali, = ali_main_loop.model.top_bricks
    gan, = gan_main_loop.model.top_bricks

    dataset = GaussianMixture(num_examples=2500,
                              means=MEANS, variances=VARIANCES, priors=None,
                              rng=None, sources=('features', 'label'))
    features, targets = dataset.indexables

    x = tensor.as_tensor_variable(features)
    z = ali.theano_rng.normal(
        size=(x.shape[0], ali.encoder.mapping.output_dim), dtype=x.dtype)
    z_hat = ali.encoder.apply(x)
    x_tilde = ali.decoder.apply(z)
    x_hat = ali.decoder.apply(z_hat)
    gan_x_tilde = gan.decoder.apply(z)

    samples = theano.function([], [x, x_tilde, x_hat, gan_x_tilde, z, z_hat])()
    x, x_tilde, x_hat, gan_x_tilde, z, z_hat = samples

    figure, axes = pyplot.subplots(nrows=2, ncols=4, figsize=(8, 4.5))
    for ax in axes.ravel():
        ax.set_aspect('equal')
    for ax in axes[0]:
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
        ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
    for ax in axes[1]:
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
    # ALI - q(x, z)
    axes[0, 0].set_title('$\mathbf{x} \sim q(\mathbf{x})$')
    axes[0, 0].scatter(x[:, 0], x[:, 1], marker='.', c=targets.ravel(),
                       alpha=0.3)
    axes[1, 1].set_title(
        '$\hat{\mathbf{z}} \sim q(\mathbf{z} \mid \mathbf{x})$')
    axes[1, 1].scatter(z_hat[:, 0], z_hat[:, 1], marker='.', c=targets.ravel(),
                       alpha=0.3)
    # ALI - p(x, z)
    axes[0, 2].set_title(
        '$\\tilde{\mathbf{x}} \sim p(\mathbf{x} \mid \mathbf{z})$')
    axes[0, 2].scatter(x_tilde[:, 0], x_tilde[:, 1], marker='.', c='black',
                       alpha=0.3)
    axes[1, 2].set_title('$\mathbf{z} \sim p(\mathbf{z})$')
    axes[1, 2].scatter(z[:, 0], z[:, 1], marker='.', c='black', alpha=0.3)
    # ALI - q(z) p(x | z) (reconstruction)
    axes[0, 1].set_title(
        '$\hat{\mathbf{x}} \sim p(\mathbf{x} \mid \hat{\mathbf{z}}$)')
    axes[0, 1].scatter(x_hat[:, 0], x_hat[:, 1], marker='.',
                       c=targets.ravel(), alpha=0.3)
    # GAN - p(x)
    axes[0, 3].set_title('GAN $\mathbf{x} = G(\mathbf{z})$')
    axes[0, 3].scatter(gan_x_tilde[:, 0], gan_x_tilde[:, 1], marker='.',
                       c='black', alpha=0.3)
    axes[1, 0].axis('off')
    axes[1, 3].axis('off')

    pyplot.tight_layout()
    if save_path is None:
        pyplot.show()
    else:
        pyplot.savefig(save_path, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot samples.")
    parser.add_argument("ali_main_loop_path", type=str)
    parser.add_argument("gan_main_loop_path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()
    with open(args.ali_main_loop_path, 'rb') as ali_src:
        with open(args.gan_main_loop_path, 'rb') as gan_src:
            main(load(ali_src), load(gan_src), args.save_path)
