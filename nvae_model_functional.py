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


class NVAE:
    CUSTOM_MODEL_CLASSES = {
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

    def __init__(self):

        self.tf_model = None
        self.input_shape = None
        self.all_merge_cells = []
        self.group_size_list = []

    def initialize(self,
                   input_shape,
                   base_num_channels,
                   num_scales,
                   num_groups,
                   num_cells,
                   num_latent,
                   num_prepost_blocks=1,
                   num_prepost_cells=2,
                   kl_center_scalar=0.0001,
                   kl_residual_scalar=0.0001,
                   use_adaptive_ngroups=True):


        self.input_shape = input_shape

        # If adaptive, reduce group sizes by factors of 2
        if not use_adaptive_ngroups:
            self.group_size_list = [num_groups] * num_scales
        else:
            self.group_size_list = [max(num_groups // 2**s, 2) for s in range(num_scales)]

            # We reverse the list because the peak of the arch is actually s=0
            self.group_size_list = self.group_size_list[::-1]

            print('Using adaptive group sizes:', self.group_size_list)

        orig_input_shape = input_shape[-3:]  # initial image shape (W, H, Ch)
        orig_side = orig_input_shape[0]
        orig_chan = orig_input_shape[-1]

        sm_scale_factor = 2 ** (num_scales + num_prepost_blocks - 1)
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
                last_cell_in_block = (i_cell == num_prepost_cells - 1)
                name = f'pre_blk{i_pre}_c{i_cell}'
                x = ResidualEncoderCell(downsample=last_cell_in_block, name=name)(x)

        #####
        # Encoder Tower -- aggregate combine_samplers after each group.
        # We treat the top of the tower as (scale, group) == (0, 0), so reverse the loops
        for s in list(range(num_scales))[::-1]:
            peak_scale = (s == 0)
            ngroups_this_scale = self.group_size_list[s]
            #print(f'Enc s={s}, ngrps={ngroups_this_scale}')

            for g in list(range(ngroups_this_scale))[::-1]:
                top_group_in_scale = (g == 0)

                for c in range(num_cells):
                    name = f'encoder_s{s}_g{g}_c{c}'
                    x = ResidualEncoderCell(name=name)(x)

                # The last encoder output gets combined with h0
                if not (top_group_in_scale and peak_scale):
                    merge_enc_left_side[s][g] = x

                if top_group_in_scale and not peak_scale:
                    name = f'encoder_s{s}_g{g}_down'
                    x = ResidualEncoderCell(downsample=True, name=name)(x)

        s_enc_peak = x

        #####
        # Decoder Tower -- aggregate combine_samplers after each group
        for s in range(num_scales):
            peak_scale = (s == 0)
            bottom_scale = (s == num_scales - 1)
            ngroups_this_scale = self.group_size_list[s]
            #print(f'Dec s={s}, ngrps={ngroups_this_scale}')

            for g in range(ngroups_this_scale):
                top_group_in_scale = (g == 0)
                last_group_in_scale = (g == ngroups_this_scale - 1)

                if top_group_in_scale and peak_scale:
                    mcell = MergeCellPeak(num_latent, h0_shape, kl_center_scalar=kl_center_scalar)
                    x = mcell(s_enc_peak)
                else:
                    s_enc = merge_enc_left_side[s][g]
                    name = f'merge_s{s}_g{g}'
                    mcell = MergeCell(num_latent,
                                      kl_center_scalar=kl_center_scalar,
                                      kl_residual_scalar=kl_residual_scalar,
                                      name=name)
                    x = mcell([s_enc, x])

                self.all_merge_cells.append(mcell)

                for c in range(num_cells):
                    name = f'decoder_s{s}_g{g}_c{c}'
                    x = ResidualDecoderCell(name=name)(x)

                if last_group_in_scale and not bottom_scale:
                    name = f'decoder_s{s}_g{g}_up'
                    x = ResidualDecoderCell(upsample=True, name=name)(x)


        #####
        # Post-processing
        for i_post in range(num_prepost_blocks):

            for i_cell in range(num_prepost_cells):
                first_cell_in_block = (i_cell == 0)
                name = f'post_blk{i_post}_c{i_cell}'
                x = ResidualDecoderCell(upsample=first_cell_in_block, name=name)(x)


        #####
        # Tail (opposite stem)
        x = tf.keras.activations.elu(x)
        x = outputs = NvaeConv2D(kernel_size=(3, 3), abs_channels=orig_chan, name='tail_stem')(x)

        model = tf.keras.Model(inputs=enc_input, outputs=outputs)
        self.tf_model = model

    def __call__(self, *args, **kwargs):
        return self.tf_model(*args, **kwargs)

    def set_merge_mode(self, new_mode, new_temperature=None):
        for layer in self.all_merge_cells:
            old_mode = layer.merge_mode
            layer.set_merge_mode(new_mode)
            if new_temperature is not None:
                    layer.set_temperature(new_temperature)

        return old_mode

    def get_last_z_output(self):
        return [cell.last_z_output for cell in self.all_merge_cells]

    def dictate_next_output(self, z_list):
        prev_merge_mode = self.set_merge_mode('dictate')
        for z, cell in zip(z_list, self.all_merge_cells):
            cell.next_z_output = z

        out = self.tf_model(tf.zeros((1,) + self.input_shape))
        self.set_merge_mode(prev_merge_mode)
        return tf.squeeze(out, axis=0)

    def blend_images(self, img1, img2, alpha_list=None):
        self.set_merge_mode('merge')
        if alpha_list is None:
            alpha_list = [0.5]

        output_is_4d = True
        if len(tf.shape(img1)) == 3:
            output_is_4d = False
            img1 = tf.expand_dims(img1, axis=0)

        if len(tf.shape(img2)) == 3:
            img2 = tf.expand_dims(img2, axis=0)

        _ = self.tf_model(img1)
        z1s = self.get_last_z_output()

        _ = self.tf_model(img2)
        z2s = self.get_last_z_output()


        z_mixtures = [self.linterp_zs(z1s, z2s, alpha) for alpha in alpha_list]
        img_mixtures = [self.dictate_next_output(zs) for zs in z_mixtures]
        if output_is_4d:
            img_mixtures = [tf.expand_dims(img, axis=0) for img in img_mixtures]

        return img_mixtures

    @staticmethod
    def linterp_zs(z_list1, z_list2, alpha):
        return [z1 * alpha + z2 * (1.0 - alpha) for z1, z2 in zip(z_list1, z_list2)]


