from unittest.mock import patch

import numpy as np
import tensorflow as tf

from configs.config import CFG
from model.model import CLICK


def dummy_load_data(*args, **kwargs):
    pass


class ClickTest(tf.test.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_normalize(self):
        pass

    def test_ouput_size(self):
        pass

    @patch('model.CLICK.DataLoader.load_data')
    def test_load_data(self, mock_data_loader):
        pass


if __name__ == '__main__':
    tf.test.main()
