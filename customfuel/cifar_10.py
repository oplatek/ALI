from os import path
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme


class Cifar10Dataset(IndexableDataset):

    def __init__(self, data_dir, is_train=True):
        if is_train:
            parts = [Cifar10Dataset._load_batch(path.join(data_dir, 'data_batch_{}'.format(i))) for i in range(1, 7)]
            self.data = {'features': np.concatenate([p['features'] for p in parts], axis=0),
                         'label': np.concatenate([p['label'] for p in parts], axis=0)}
        else:
            self.data = Cifar10Dataset._load_batch(path.join(data_dir, 'test_batch'))

        super(Cifar10Dataset, self).__init__(self.data)
        self.is_train = is_train

    @staticmethod
    def _load_batch(data_path):
        dd = pickle.load(open(data_path, 'rb'))
        return {'features': (np.array(dd['data']) / 255.0).astype('float32').reshape((-1, 3, 32, 32)),
                'label': np.array(dd['labels']).reshape((-1,)).astype('int32')}

    def get_stream(self, batch_size):
        if self.is_train:
            iter_scheme = ShuffledScheme(self.num_examples, batch_size)
        else:
            iter_scheme = SequentialScheme(self.num_examples, batch_size)
        return DataStream(self, iteration_scheme=iter_scheme)


if __name__ == '__main__':
    dataset = Cifar10Dataset('../data/cifar10', False)
    for d in dataset.get_stream(8).get_epoch_iterator(as_dict=True):
        print d.keys()
        print d.values()
        break

    dataset = Cifar10Dataset('../data/cifar10', True)
    for d in dataset.get_stream(8).get_epoch_iterator(as_dict=True):
        print d.keys()
        break

