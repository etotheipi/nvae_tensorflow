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
from nvae_layers import KLDivergence

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
                enc_x = merge_enc_left_side[s][g]
                merge_sampled = CombinerSampler(kl_loss_scalar, name=f'combine_{s}_{g}')([enc_x, x])
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
        kl_center_scalar=0.0001,
        kl_residual_scalar=0.01,
        use_adaptive_ngroups=True):

    all_merge_cells = []

    # If adaptive, reduce group sizes by factors of 2
    if not use_adaptive_ngroups:
        group_size_list = [ngroups] * nscales
    else:
        group_size_list = [max(ngroups // 2**s, 2) for s in range(nscales)]

        # We reverse the list because the peak of the arch is actually s=0
        group_size_list = group_size_list[::-1]

    print('Group sizes', group_size_list)


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

    merge_enc_left_side = defaultdict(dict)

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
        ngroups_this_scale = group_size_list[s]
        print(f'Enc s={s}, ngrps={ngroups_this_scale}')
        
        for g in list(range(ngroups_this_scale))[::-1]:
            top_group_in_scale = (g == 0)
            
            for c in range(ncells):
                #last_cell = (c == ncells - 1)
                
                name = f'encoder_s{s}_g{g}_c{c}'
                x = ResidualEncoderCell(name=name)(x)

            # The last encoder output gets combined with h0
            if not (top_group_in_scale and peak_scale):
                merge_enc_left_side[s][g] = x
                    
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
        ngroups_this_scale = group_size_list[s]
        print(f'Dec s={s}, ngrps={ngroups_this_scale}')

        for g in range(ngroups_this_scale):
            top_group_in_scale = (g == 0)
            last_group_in_scale = (g == ngroups_this_scale - 1)
            
            if top_group_in_scale and peak_scale:
                mcell = MergeCellPeak(nlatent, h0_shape, kl_center_scalar=kl_center_scalar)
                x = mcell(s_enc_peak)
            else:
                s_enc = merge_enc_left_side[s][g]
                name = f'merge_s{s}_g{g}'                                              
                mcell = MergeCell(nlatent,
                                  kl_center_scalar=kl_center_scalar,
                                  kl_residual_scalar=kl_residual_scalar,
                                  name=name)
                x = mcell([s_enc, x])

            all_merge_cells.append(mcell)

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
    model.merge_cells = all_merge_cells
    
    return model


def set_model_merge_mode(model, new_mode, new_temperature=None):
    for layer in model.layers:
        try:
            old_mode = layer.merge_mode
            layer.set_merge_mode(new_mode)
        except Exception as e:
            pass
    
    if new_temperature is not None:
        for layer in model.layers:
            try:
                layer.set_temperature(new_temperature)
            except Exception as e:
                pass
        
    return old_mode

def get_last_z_output(model):
    return [cell.last_z_output for cell in model.merge_cells]


def dictate_next_output(model, z_list, input_shape):
    prev_merge_mode = set_model_merge_mode(model, 'dictate')
    for z, cell in zip(z_list, model.merge_cells):
        cell.next_z_output = z
        
    out = model(tf.zeros((1,) + input_shape))
    set_model_merge_mode(model, prev_merge_mode)
    return out


def linterp_zs(z_list1, z_list2, alpha):
    return [z1 * alpha + z2 * (1.0 - alpha) for z1, z2 in zip(z_list1, z_list2)]


custom_model_classes = {
    'SqueezeExciteLayer': SqueezeExciteLayer,
    'NvaeConv2D': NvaeConv2D,
    'ResidualDecoderCell': ResidualDecoderCell,
    'ResidualEncoderCell': ResidualEncoderCell,
    'FactorizedDownsample': FactorizedDownsample,
    'KLDivergence': KLDivergence,
    'Sampling': Sampling,
    'MergeCellPeak': MergeCellPeak,
    'MergeCell': MergeCell,
}
