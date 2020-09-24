from collections import defaultdict
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa

from nvae_layers import SqueezeExciteLayer
from nvae_layers import FactorizedDownsample
from nvae_layers import ResidualDecoderCell
from nvae_layers import ResidualEncoderCell
from nvae_layers import MergeCellPeak
from nvae_layers import MergeCell
from nvae_layers import NvaeConv2D
from nvae_layers import Sampling

class NVAE(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 base_num_channels,
                 nscales,
                 ngroups,
                 ncells,
                 nlatent,
                 npreblocks=1,
                 nprecells=2,
                 *args,
                 **kwargs):
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
        self.peak_side = self.orig_side // sm_scale_factor
        self.peak_chan = self.base_num_channels * sm_scale_factor

        # This is the learnable parameter at the peak between encoder & decoder
        self.h_peak = self.add_weight(shape=(self.peak_side, self.peak_side, self.peak_chan))


        self.tower = {}

        self.tower['stem'] = [
            NvaeConv2D(kernel_size=(3, 3), abs_channels=base_num_channels, name='pre_stem')
        ]

        self.tower['preproc'] = []
        for i_pre in range(npreblocks):
            for i_cell in range(nprecells):
                last_in_block = (i_cell == nprecells - 1)
                name = f'pre_blk{i_pre}_c{i_cell}'
                res_cell = ResidualEncoderCell(downsample=last_in_block, name=name)
                self.tower['preproc'].append(res_cell)


        #####
        # Encoder Tower -- aggregate combine_samplers after each group.
        # We treat the top of the tower as (scale, group) == (0, 0), so reverse the loops
        self.tower['encoder'] = []
        self.tower['merge_levels'] = {}
        for s in list(range(nscales))[::-1]:
            peak_scale = (s == 0)

            for g in list(range(ngroups))[::-1]:
                top_group_in_scale = (g == 0)

                for c in range(ncells):
                    name = f'encoder_s{s}_g{g}_c{c}'
                    self.tower['encoder'].append(ResidualEncoderCell(name=name))

                if top_group_in_scale:
                    if not peak_scale:
                        self.tower['merge_levels'][s*ngroups+g] = MergeCellPeak()
                    else:
                        name = f'encoder_s{s}_g{g}_down'
                        res_down_cell = ResidualEncoderCell(downsample=True, name=name)
                        self.tower['encoder'].append(res_down_cell)
                        self.tower['merge_levels'][s*ngroups+g] = MergeCell()

        #####
        # Encoder0 -- named this way in the NVLabs code
        x = tf.keras.activations.elu(x)
        x = NvaeConv2D(kernel_size=(1, 1), scale_channels=2, name='encoder_peak')(x)
        x = tf.keras.activations.relu(x)

        sample0 = Sampling()(x)
        h0 = tf.constant(tf.zeros(shape=h0_shape), name='h_peak')

        x = h0 + sample0
        # x = sample0

        #####
        # Decoder Tower -- aggregate combine_samplers after each group
        for s in range(nscales):
            peak_scale = (s == 0)

            for g in range(ngroups):
                top_group_in_scale = (g == 0)

                if peak_scale and top_group_in_scale:
                    continue

                for c in range(ncells):
                    # last_cell = c == ncells - 1
                    name = f'decoder_s{s}_g{g}_c{c}'
                    x = ResidualDecoderCell(name=name)(x)

                # Get the combiner
                i_merge = s * ngroups + g
                enc_x = merge_enc_left_side[i_merge]
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
        # Tail (opposite stem)
        x = tf.keras.activations.elu(x)
        x = outputs = NvaeConv2D(kernel_size=(3, 3), abs_channels=orig_chan, name='tail_stem')(x)


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
    h0_shape = (h0_side, h0_side, h0_chan)

    merge_enc_left_side = {}

    # Let's get this party started
    x = enc_input = L.Input(input_shape)

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
        peak_scale = (s == 0)
        
        for g in list(range(ngroups))[::-1]:
            top_group_in_scale = (g == 0)
            
            for c in range(ncells):
                #last_cell = (c == ncells - 1)
                
                name = f'encoder_s{s}_g{g}_c{c}'
                x = ResidualEncoderCell(name=name)(x)

            # The last encoder output gets combined with h0
            if not (top_group_in_scale and peak_scale):
                merge_enc_left_side[s*ngroups + g] = x
                    
            if top_group_in_scale and not peak_scale:
                name = f'encoder_s{s}_g{g}_down'
                x = ResidualEncoderCell(downsample=True, name=name)(x)
                chan = chan * 2
                side = side // 2
    
    # Can we use variables within functional models?  Well what we want is just a 
    #h0 = tf.Variable(shape=h0_shape, trainable=True, initial_value=tf.random.normal(h0_shape), name='h_peak')
    s_enc_peak = x
    #h0 = tf.constant(tf.zeros(shape=h0_shape))
    
    #####
    # Decoder Tower -- aggregate combine_samplers after each group
    for s in range(nscales):
        peak_scale = (s == 0)
        bottom_scale = (s == nscales - 1)
        
        for g in range(ngroups):
            top_group_in_scale = (g == 0)
            last_group_in_scale = (g == ngroups - 1)
            
            if top_group_in_scale and peak_scale:
                x = MergeCellPeak(nlatent, peak_shape=h0_shape)(s_enc_peak)
            else:
                i_merge = s*ngroups + g
                s_enc = merge_enc_left_side[i_merge]
                name = f'merge_s{s}_g{g}'                                              
                x = MergeCell(nlatent, name=name)([s_enc, x])
                                                                  
            for c in range(ncells):
                name = f'decoder_s{s}_g{g}_c{c}'
                x = ResidualDecoderCell(name=name)(x)
                    
            if last_group_in_scale and not bottom_scale:
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
            name = f'post_blk{i_post}_c{i_cell}'
            x = ResidualDecoderCell(upsample=first_cell_in_block, name=name)(x)
            
            
    #####
    # Tail (opposite stem)
    x = tf.keras.activations.elu(x)
    x = outputs = NvaeConv2D(kernel_size=(3, 3), abs_channels=orig_chan, name='tail_stem')(x)
    
    model = tf.keras.Model(inputs=enc_input, outputs=outputs)
    
    return model
    
    
    