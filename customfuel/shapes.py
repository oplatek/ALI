from os import path
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import defaultdict

import numpy as np
import numpy.random as npr

from PIL import Image, ImageDraw
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme


class ShapesDataset(IndexableDataset):

    def __init__(self, num_examples, img_size, min_diameter, seed=12345):

        npr.seed(seed)

        self.data = defaultdict(list)
        for _ in range(num_examples):
            diameter = npr.randint(min_diameter, 3 * img_size / 2 / 4)
            bg_color = tuple(npr.randint(0, 3 * 255, size=3))
            front_color = tuple(npr.randint(0, 3 * 255, size=3))

            label = npr.randint(2) == 0
            if label == 0:
                features = ShapesDataset.draw_cross(img_size, diameter, bg_color, front_color, width=2)
            else:
                features = ShapesDataset.draw_circle(img_size, diameter, bg_color, front_color)
            self.data['features'].append(features.astype('float32'))
            self.data['labels'].append(label)

        self.data['features'] = np.array(self.data['features']).transpose(0, 3, 1, 2)
        self.data['labels'] = np.array(self.data['features'])

        super(ShapesDataset, self).__init__(self.data)

    def create_stream(self, batch_size, is_train):
        if is_train:
            iter_scheme = ShuffledScheme(self.num_examples, batch_size)
        else:
            iter_scheme = SequentialScheme(self.num_examples, batch_size)
        stream = DataStream(self, iteration_scheme=iter_scheme)

        # for d in stream.get_epoch_iterator(as_dict=True):
        #     for k, v in d.iteritems():
        #         print('{}:\t{},\t{}'.format(k, v.dtype, v.shape))
        #     break

        return stream

    @staticmethod
    def draw_cross(img_size, diameter, bg_color, cross_color, width):
        center_x, center_y = npr.randint(diameter, img_size-diameter+1, size=2)

        im = Image.new('RGB', (img_size, img_size), color=bg_color)

        draw = ImageDraw.Draw(im)
        draw.line((center_x, center_y-diameter, center_x, center_y+diameter), fill=cross_color, width=width)
        draw.line((center_x-diameter, center_y, center_x+diameter, center_y), fill=cross_color, width=width)

        return np.asarray(im, dtype=np.uint8)

    @staticmethod
    def draw_circle(img_size, diameter, bg_color, circle_color):
        center_x, center_y = npr.randint(diameter, img_size-diameter+1, size=2)

        im = Image.new('RGB', (img_size, img_size), color=bg_color)

        draw = ImageDraw.Draw(im)
        draw.ellipse((center_x-diameter, center_y-diameter, center_x+diameter, center_y+diameter), fill=circle_color)

        return np.asarray(im, dtype=np.uint8)


if __name__ == '__main__':

    train_dataset = ShapesDataset(num_examples=20, img_size=32, min_diameter=3, seed=1234)
    train_stream = train_dataset.create_stream(batch_size=2, is_train=True)

    for d in train_stream.get_epoch_iterator(as_dict=True):
        print(d)

    valid_dataset = ShapesDataset(num_examples=10, img_size=320, min_diameter=30, seed=5678)
    valid_stream = valid_dataset.create_stream(batch_size=2, is_train=False)

    for d in train_stream.get_epoch_iterator(as_dict=True):
        print(d)
