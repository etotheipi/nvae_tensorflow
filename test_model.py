import os
import unittest
from nvae_layers import *
from nvae_model_subclass import NVAE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

class TestModelCreation(unittest.TestCase):

    def test_create_nvae(self):
        test_model = NVAE(
            input_shape=(32, 32, 3),
            base_num_channels=8,
            num_scales=2,
            num_groups=2,
            num_cells=2,
            num_latent=2,
            num_prepost_blocks=1,
            num_prepost_cells=2)
        
        self.assertEqual((4, 32, 32, 3), test_model(tf.zeros(shape=(4, 32, 32, 3))).shape)
                         
    def test_no_prepost_step(self):
        test_model = NVAE(
            input_shape=(32, 32, 3),
            base_num_channels=16,
            nscales=2,
            ngroups=2,
            ncells=2,
            nlatent=2,
            num_prepost_blocks=0,
            num_prepost_cells=0)
        
        
        self.assertEqual((32, 32, 3), test_model.layers[-1].output_shape[1:])
        
    def test_one_scale(self):
        test_model=NVAE(
            input_shape=(32, 32, 3),
            base_num_channels=16,
            nscales=1,
            ngroups=2,
            ncells=2,
            nlatent=2,
            num_prepost_blocks=0,
            num_prepost_cells=0)
        self.assertEqual((32, 32, 3), test_model.layers[-1].output_shape[1:])
        
    def test_one_group(self):
        test_model=NVAE(
            input_shape=(32, 32, 3),
            base_num_channels=16,
            nscales=2,
            ngroups=1,
            ncells=2,
            nlatent=2,
            num_prepost_blocks=0,
            num_prepost_cells=0)
        self.assertEqual((32, 32, 3), test_model.layers[-1].output_shape[1:])

        
if __name__ == '__main__':
    unittest.main()
