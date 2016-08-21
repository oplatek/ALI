#!/usr/bin/env python
import argparse
from collections import defaultdict
import cPickle as pickle

import numpy as np
import pandas as pd
import theano
from theano import tensor as tt
from blocks.serialization import load

from customfuel import Cifar10Dataset


def main(main_loop, save_path, batch_size=1024):

    print('Loading & compilign ALI')
    ali, = main_loop.model.top_bricks
    x = tt.tensor4('features')
    encode = theano.function([x], ali.encoder.apply(x))

    print('Creating dataset')
    dataset = Cifar10Dataset('/home/belohlavek/data/cifar10', is_train=False)
    stream = dataset.get_stream(batch_size=batch_size)

    print('Main loop')
    result = defaultdict(list)
    for bid, batch in enumerate(stream.get_epoch_iterator(as_dict=True)):
        print('\tProcessing batch #{}/{}'.format(bid, dataset.num_examples/batch_size))

        examples = np.array(batch['features']).astype('float32')
        states = encode(examples).reshape((len(examples), -1))

        result['labels'].extend(batch['label'])
        result['states'].extend(states)
    
    print('Saving')
    result = pd.DataFrame.from_dict(result)
    pickle.dump(result, open(save_path, 'wb'))

    print('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reconstructions.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the reconstructions.")
    args = parser.parse_args()

    print('Opening main_loop')
    with open(args.main_loop_path, 'rb') as src:
        print('Loading main_loop')
        main_loop = load(src)
        main(main_loop, args.save_path)
