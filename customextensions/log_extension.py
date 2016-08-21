from blocks.extensions import SimpleExtension
from pandas import DataFrame


class LogExtension(SimpleExtension):
    def __init__(self, log_file, **kwargs):
        kwargs.setdefault('before_first_epoch', True)
        kwargs.setdefault('after_training', True)
        kwargs.setdefault('after_epoch', True)

        self.log_file = log_file
        super(LogExtension, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        DataFrame.from_dict(self.main_loop.log, orient='index').to_csv(self.log_file)
