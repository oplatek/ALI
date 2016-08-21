#!/usr/bin/env python
import logging
from argparse import ArgumentParser

import theano
from theano import tensor as tt

from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import MLP, Tanh, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import CIFAR10
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.roles import WEIGHT

from customfuel import Cifar10Dataset
from customextensions import LogExtension


def main(save_to, num_epochs, batch_size):
    mlp = MLP([Tanh(), Tanh(), Tanh(), Softmax()], [3072, 4096, 1024, 512, 10],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0))
    mlp.initialize()
    x = tt.tensor4('features', dtype='float32')
    y = tt.vector('label', dtype='int32')

    probs = mlp.apply(x.reshape((-1,3072)))
    cost = CategoricalCrossEntropy().apply(y, probs)
    error_rate = MisclassificationRate().apply(y, probs)

    cg = ComputationGraph([cost])
    ws = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + .00005 * sum(([(w**2).sum() for w in ws]))
    cost.name = 'final_cost'

    train_dataset = Cifar10Dataset(data_dir='/home/belohlavek/data/cifar10', is_train=True)
    valid_dataset = Cifar10Dataset(data_dir='/home/belohlavek/data/cifar10', is_train=False)

    train_stream = train_dataset.get_stream(batch_size)
    valid_stream = valid_dataset.get_stream(batch_size)

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Adam(learning_rate=0.001))
    extensions = [Timing(),
                  LogExtension('/home/belohlavek/ALI/mlp.log'),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring([cost, error_rate], valid_stream, prefix="test"),
                  TrainingDataMonitoring(
                      [cost, error_rate, aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  Printing()]

    main_loop = MainLoop(algorithm,
                         train_stream,
                         model=Model(cost),
                         extensions=extensions)

    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("CIFAR10")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs to do.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("save_to", default="cifar.pkl", nargs="?",
                        help=("Destination to save the state of the training process."))
    args = parser.parse_args()
    main(args.save_to, args.num_epochs, args.batch_size)
