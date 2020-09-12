from collections import defaultdict
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa

from nvae_layers import SqueezeExciteLayer
from nvae_layers import FactorizedDownsample
from nvae_layers import ResidualDecoderCell
from nvae_layers import ResidualEncoderCell
from nvae_layers import CombinerSampler
from nvae_layers import NvaeConv2D
from nvae_layers import Sampling

class NVAE(tf.keras.Model):
    def __init__(self, input_shape, base_num_channels, nscales, ngroups, nlatent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_scales = nscales
        self.groups_per_scale = ngroups
        self.latent_per_group = nlatent

        assert input_shape[-3] != input_shape[-2], 'Image shape must be square'
        self.input_shape = input_shape[-3:]  # initial image shape (W, H, Ch)
        self.orig_side = input_shape[-2]  # initial image shape (W)
        self.orig_chan = input_shape[-1]  # initial num channels (Ch)

        self.base_num_channels = base_num_channels

        sm_scale_factor = 2 ** (self.num_scales - 1)
        assert self.orig_side % sm_scale_factor == 0, f'Image size must be multiple of {sm_scale_factor}'
        self.smallest_scale = self.orig_side // sm_scale_factor
        self.largest_num_chan = self.base_num_channels * sm_scale_factor

        # This is the learnable parameter at the peak between encoder & decoder
        self.h_peak = self.add_weight(shape=(self.smallest_scale, self.smallest_scale, self.largest_num_chan))

        self.encoder_blocks = []
        self.decoder_blocks = []
        self.decoder_blocks = []
        # STUB -- going to do the functional version below, first
        



def create_nvae(
    input_shape,
    base_num_channels,
    nscales,
    ngroups,
    ncells,
    nlatent,
    num_prepost_blocks=2,
    num_prepost_cells=2,
    kl_loss_scalar=0.001):

    # 
    orig_input_shape = input_shape[-3:]  # initial image shape (W, H, Ch)
    side = orig_side = orig_input_shape[0]
    orig_chan = orig_input_shape[-1]
    chan = base_num_channels
    
    sm_scale_factor = 2 ** (nscales + num_prepost_blocks - 1)
    assert orig_side % sm_scale_factor == 0, f"Original image size must be multiple of {sm_scale_factor}"
    h0_side = orig_side // sm_scale_factor
    h0_chan = base_num_channels * sm_scale_factor
    h0_shape = (1, h0_side, h0_side, h0_chan)

    encoder_merge_inputs = {}

    # Let's get this party started
    x = inputs = L.Input(input_shape)

    #####
    # Stem -- Expand from 3 channels to num_channels_enc
    x = NvaeConv2D(kernel_size=(3, 3), abs_channels=base_num_channels, name='pre_stem')(x)

    #####
    # Preprocessing -- No combiner/latent outputs
    for i_pre in range(num_prepost_blocks):
        for i_cell in range(num_prepost_cells):
            last_cell_in_block = (i_cell == num_prepost_cells -1)
            name = f'pre_blk{i_pre}_c{i_cell}'
            x = ResidualEncoderCell(downsample=last_cell_in_block, name=name)(x)
            
        chan = chan * 2
        side = side // 2

    #####
    # Encoder Tower -- aggregate combine_samplers after each group. 
    # We treat the top of the tower as (scale, group) == (0, 0), so reverse the loops
    for s in list(range(nscales))[::-1]:
        last_scale = (s == 0)
        
        for g in list(range(ngroups))[::-1]:
            last_group = (g == 0)
            
            for c in range(ncells):
                #last_cell = (c == ncells - 1)
                
                name = f'encoder_s{s}_g{g}_c{c}'
                x = ResidualEncoderCell(name=name)(x)

            # The last encoder output gets combined with h0
            if not (last_group and last_scale):
                encoder_merge_inputs[s*ngroups + g] = x
                    
            if last_group and not last_scale:
                name = f'encoder_s{s}_g{g}_down'
                x = ResidualEncoderCell(downsample=True, name=name)(x)
                chan = chan * 2
                side = side // 2
    

    #####
    # Encoder0 -- named this way in the NVLabs code
    x = tf.keras.activations.elu(x)
    x = NvaeConv2D(kernel_size=(1, 1), scale_channels=2, name='encoder_peak')(x)
    x = tf.keras.activations.relu(x)
    
    sample0 = Sampling()(x)
    
    # Can we use variables within functional models?  Well what we want is just a 
    h0 = tf.Variable(shape=h0_shape, trainable=True, initial_value=tf.random.normal(h0_shape), name='h_peak')
    
    x = h0 + sample0
    
    #####
    # Decoder Tower -- aggregate combine_samplers after each group
    for s in range(nscales):
        last_scale = (s == 0)
        
        for g in range(ngroups):
            last_group = (g == 0)
            
            if last_scale and last_group:
                continue
            
            for c in range(ncells):
                #last_cell = c == ncells - 1
                name = f'decoder_s{s}_g{g}_c{c}'
                x = ResidualDecoderCell(name=name)(x)
                
            # Get the combiner
            i_merge = s*ngroups + g
            enc_x = encoder_merge_inputs[i_merge]
            merge_sampled = CombinerSampler(kl_loss_scalar, name=f'combine_{i_merge}')([enc_x, x])
            x = merge_sampled + x
                    
            if g == ngroups - 1 and s != nscales - 1:
                name = f'decoder_s{s}_g{g}_up'
                x = ResidualDecoderCell(upsample=True, name=name)(x)
                chan = chan // 2
                side = side * 2
    
    
    #####
    # Post-processing
    for i_post in range(num_prepost_blocks):
        
        chan = chan * 2
        side = side // 2
        
        for i_cell in range(num_prepost_cells):
            first_cell_in_block = (i_cell == 0)
            name = f'post_blk{i_pre}_c{i_cell}'
            x = ResidualDecoderCell(upsample=first_cell_in_block, name=name)(x)
            
            
    #####
    # Post-stem
    x = tf.keras.activations.elu(x)
    x = outputs = NvaeConv2D(kernel_size=(3, 3), abs_channels=orig_chan, name='post_stem')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.h0 = h0
    
    return model
    
    
    