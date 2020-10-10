import os
import unittest
from nvae_layers import *
from nvae_model import create_nvae
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

class TestLayerSizes(unittest.TestCase):
    def test_nvae_conv2d_sizes(self):
        x = inputs = L.Input(shape=(32, 32, 3))
        x = NvaeConv2D(kernel_size=3, abs_channels=32)(x)
        x = NvaeConv2D(kernel_size=3)(x)
        x = NvaeConv2D(kernel_size=3)(x)
        x = NvaeConv2D(kernel_size=1, downsample=True, scale_channels=2)(x)
        x = NvaeConv2D(kernel_size=3)(x)
        x = NvaeConv2D(kernel_size=3)(x)
        x = NvaeConv2D(kernel_size=1, downsample=True, scale_channels=2)(x)
        x = NvaeConv2D(kernel_size=3)(x)
        x = NvaeConv2D(kernel_size=3)(x)
        x = NvaeConv2D(kernel_size=1, upsample=True, scale_channels=-2)(x)
        x = NvaeConv2D(kernel_size=3)(x)
        x = NvaeConv2D(kernel_size=3)(x)
        x = NvaeConv2D(kernel_size=1, upsample=True, scale_channels=-2)(x)
        x = NvaeConv2D(kernel_size=1, abs_channels=3)(x)

        test_model = tf.keras.Model(inputs=inputs, outputs=x)
        self.assertEqual((32, 32, 3), test_model.layers[-1].output_shape[1:])
                         
            
    def test_residual_sizes(self):
        x = inputs = L.Input(shape=(32, 32, 3))
        x = NvaeConv2D(kernel_size=3, abs_channels=32)(x)
        x = ResidualEncoderCell()(x)
        x = ResidualEncoderCell()(x)
        x = ResidualEncoderCell(downsample=True)(x)
        x = ResidualEncoderCell()(x)
        x = ResidualEncoderCell()(x)
        x = ResidualEncoderCell(downsample=True)(x)
        x = ResidualDecoderCell()(x)
        x = ResidualDecoderCell()(x)
        x = ResidualDecoderCell(upsample=True)(x)
        x = ResidualDecoderCell()(x)
        x = ResidualDecoderCell()(x)
        x = ResidualDecoderCell(upsample=True)(x)
        x = NvaeConv2D(kernel_size=3, abs_channels=3)(x)

        test_model = tf.keras.Model(inputs=inputs, outputs=x)
        self.assertEqual((32, 32, 3), test_model.layers[-1].output_shape[1:])

    def test_create_nvae(self):
        create_nvae(
            input_shape=(32, 32, 3),
            base_num_channels=8,
            nscales=2,
            ngroups=2,
            ncells=2,
            nlatent=2,
            num_prepost_blocks=1,
            num_prepost_cells=2)
        
        
        self.assertEqual((32, 32, 3), test_model.layers[-1].output_shape[1:])
                         
    def test_no_prepost_step(self):
        create_nvae(
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
        create_nvae(
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
        create_nvae(
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

    
    
    