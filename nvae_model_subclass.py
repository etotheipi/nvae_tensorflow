# This particular implementation didn't work out.  The layer losses didn't propagate up to
# the model, and the graph of the layers is
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
                 num_scales,
                 num_groups,
                 num_cells,
                 num_latent,
                 num_prepost_blocks=1,
                 num_prepost_cells=2,
                 kl_center_scalar=0.0001,
                 kl_residual_scalar=0.0001,
                 use_adaptive_ngroups=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.num_scales = num_scales
        self.groups_per_scale = num_groups
        self.cells_per_group = num_cells
        self.latent_per_group = num_latent
        self.num_prepost_blocks = num_prepost_blocks
        self.num_prepost_cells = num_prepost_cells
        self.merge_mode = 'merge'

        print(input_shape)
        assert input_shape[-3] == input_shape[-2], 'Image shape must be square'
        self.orig_input_shape = input_shape[-3:]  # initial image shape (W, H, Ch)
        self.orig_side = input_shape[-2]  # initial image shape (W)
        self.orig_chan = input_shape[-1]  # initial num channels (Ch)

        self.base_num_channels = base_num_channels

        sm_scale_factor = 2 ** (self.num_scales + self.num_prepost_blocks - 1)
        assert self.orig_side % sm_scale_factor == 0, f'Image size must be multiple of {sm_scale_factor}'
        self.peak_side = self.orig_side // sm_scale_factor
        self.peak_chan = self.base_num_channels * sm_scale_factor
        self.peak_shape = (self.peak_side, self.peak_side, self.peak_chan)

        # If adaptive, reduce group sizes by factors of 2
        self.use_adaptive_ngroups = use_adaptive_ngroups
        if not self.use_adaptive_ngroups:
            self.group_size_list = [num_groups] * num_scales
        else:
            self.group_size_list = [max(num_groups // 2 ** s, 2) for s in range(num_scales)]

            # We reverse the list because the peak of the arch is actually s=0
            self.group_size_list = self.group_size_list[::-1]

        print('Group sizes', self.group_size_list)

        self.last_encoder_outputs = None

        # The .tower var holds all layers in a dictionary
        self.tower = {}

        #####
        self.tower['stem'] = NvaeConv2D(
            kernel_size=(3, 3),
            abs_channels=base_num_channels,
            name='pre_stem')

        #####
        self.tower['preproc'] = []
        for i_pre in range(self.num_prepost_blocks):
            for i_cell in range(self.num_prepost_cells):
                last_in_block = (i_cell == self.num_prepost_cells - 1)
                name = f'pre_blk{i_pre}_c{i_cell}'
                res_cell = ResidualEncoderCell(downsample=last_in_block, name=name)
                self.tower['preproc'].append(res_cell)


        #####
        # Encoder Tower -- aggregate combine_samplers after each group.
        # We treat the top of the tower as (scale, group) == (0, 0), so reverse the loops
        self.tower['encoder'] = []
        self.tower['merge_levels'] = defaultdict(dict)
        for s in list(range(self.num_scales))[::-1]:
            peak_scale = (s == 0)
            ngroups_this_scale = self.group_size_list[s]
            for g in list(range(ngroups_this_scale))[::-1]:
                top_group_in_scale = (g == 0)

                for c in range(self.cells_per_group):
                    name = f'encoder_s{s}_g{g}_c{c}'
                    self.tower['encoder'].append(ResidualEncoderCell(name=name))

                # Between scales we add an extra downsampling cell
                if top_group_in_scale and not peak_scale:
                    name = f'encoder_s{s}_g{g}_down'
                    self.tower['encoder'].append(ResidualEncoderCell(downsample=True, name=name))

                # Every group has a merge level
                if top_group_in_scale and peak_scale:
                    self.tower['merge_levels'][s][g] = MergeCellPeak(
                        self.latent_per_group,
                        self.peak_shape,
                        kl_center_scalar=kl_center_scalar)
                else:
                    self.tower['merge_levels'][s][g] = MergeCell(
                        self.latent_per_group,
                        kl_center_scalar=kl_center_scalar,
                        kl_residual_scalar=kl_residual_scalar)



        #####
        # Decoder Tower -- aggregate combine_samplers after each group
        self.tower['decoder'] = []
        for s in range(self.num_scales):
            peak_scale = (s == 0)
            bottom_scale = (s == self.num_scales - 1)
            ngroups_this_scale = self.group_size_list[s]
            #print(f'Dec s={s}, ngrps={ngroups_this_scale}')

            for g in range(ngroups_this_scale):
                top_group_in_scale = (g == 0)
                last_group_in_scale = (g == ngroups_this_scale - 1)

                for c in range(self.cells_per_group):
                    name = f'decoder_s{s}_g{g}_c{c}'
                    self.tower['decoder'].append(ResidualDecoderCell(name=name))

                if last_group_in_scale and not bottom_scale:
                    name = f'decoder_s{s}_g{g}_up'
                    self.tower['decoder'].append(ResidualDecoderCell(upsample=True, name=name))

        #####
        # Post-processing
        self.tower['postproc'] = []
        for i_post in range(self.num_prepost_blocks):
            for i_cell in range(self.num_prepost_cells):
                first_cell_in_block = (i_cell == 0)
                name = f'post_blk{i_post}_c{i_cell}'
                res_cell = ResidualDecoderCell(upsample=first_cell_in_block, name=name)
                self.tower['postproc'].append(res_cell)

        #####
        # Tail (opposite stem)
        self.tower['tail'] = NvaeConv2D(kernel_size=(3, 3), abs_channels=self.orig_chan, name='tail_stem')

    def set_merge_mode(self, new_merge_mode):
        assert new_merge_mode in ['merge', 'sample', 'dictate']
        self.merge_mode = new_merge_mode

        for _, glist in self.tower['merge_levels'].items():
            for _, layer in glist.items():
                layer.set_merge_mode('merge')

    def call_encoder(self, img_inputs, training=False):
        #self.set_merge_mode('merge')

        encoder_side_outputs = defaultdict(dict)

        x = img_inputs
        x = self.tower['stem'](x, training=training)

        for layer in self.tower['preproc']:
            x = layer(x, training=training)

        #####
        # Encoder Tower -- aggregate combine_samplers after each group.
        # We treat the top of the tower as (scale, group) == (0, 0), so reverse the loops
        linear_cell_index = 0
        for s in list(range(self.num_scales))[::-1]:
            peak_scale = (s == 0)
            ngroups_this_scale = self.group_size_list[s]
            for g in list(range(ngroups_this_scale))[::-1]:
                top_group_in_scale = (g == 0)

                for c in range(self.cells_per_group):
                    x = self.tower['encoder'][linear_cell_index](x, training=training)
                    linear_cell_index += 1

                encoder_side_outputs[s][g] = x

                if top_group_in_scale and not peak_scale:
                    # There's an extra downsample cell at the end of each scale (except peak)
                    x = self.tower['encoder'][linear_cell_index](x)
                    linear_cell_index += 1

        return encoder_side_outputs


    def call_decoder(self, encoder_inputs, training=False):
        prev_merge_mode = self.merge_mode
        if encoder_inputs is None:
            self.set_merge_mode('sample')

        #####
        # Decoder Tower -- aggregate combine_samplers after each group
        linear_cell_index = 0
        x = None
        for s in range(self.num_scales):
            peak_scale = (s == 0)
            bottom_scale = (s == self.num_scales - 1)
            ngroups_this_scale = self.group_size_list[s]

            for g in range(ngroups_this_scale):
                top_group_in_scale = (g == 0)
                last_group_in_scale = (g == ngroups_this_scale - 1)
                merge_cell = self.tower['merge_levels'][s][g]

                s_enc = encoder_inputs[s][g]

                if top_group_in_scale and peak_scale:
                    # Peak merge cell only takes one input (merges with trainable param)
                    x = merge_cell(s_enc, training=training)
                else:
                    # Otherwise, merge previous output with s_enc
                    x = merge_cell([s_enc, x], training=training)

                for c in range(self.cells_per_group):
                    x = self.tower['decoder'][linear_cell_index](x, training=training)
                    linear_cell_index += 1

                if last_group_in_scale and not bottom_scale:
                    x = self.tower['decoder'][linear_cell_index](x, training=training)
                    linear_cell_index += 1

        #####
        # Post-processing
        for layer in self.tower['postproc']:
            x = layer(x, training=training)

        #####
        # Tail (opposite stem)
        x = self.tower['tail'](x, training=training)

        self.set_merge_mode(prev_merge_mode)
        return x


    def call(self, inputs, training=False):
        """
        Roundtrip from img to reconstruction
        """
        prev_merge_mode = self.merge_mode
        self.set_merge_mode('merge')
        enc_outputs = self.call_encoder(inputs, training)
        dec_outputs = self.call_decoder(enc_outputs, training)
        self.set_merge_mode(prev_merge_mode)

        return dec_outputs

    def sample(self):
        prev_merge_mode = self.merge_mode
        self.set_merge_mode('sample')
        dec_outputs = self.call_decoder(encoder_inputs=None, training=False)
        self.set_merge_mode(prev_merge_mode)
        return dec_outputs

    def decode_dictate(self, encoder_side_outputs):
        prev_merge_mode = self.merge_mode
        self.set_merge_mode('dictate')

        dec_outputs = self.call_decoder(encoder_side_outputs, training=False)
        self.set_merge_mode(prev_merge_mode)

        return dec_outputs

    @staticmethod
    def blend_encoder_outputs(enc1, enc2, alpha=0.5):
        """
        :param enc1:  Outputs of call_encode() for img1
        :param enc2:  Outputs of call_encode() for img2
        :param alpha: Blend coefficient - 0.8 means 80% img1, 20% img2
        :return:
        """
        outputs = defaultdict(dict)
        scales1 = set(enc1.keys())
        scales2 = set(enc2.keys())
        assert len(scales1.symmetric_difference(scales2)) == 0, "Input has different scales"

        for s in scales1:
            groups1 = set(enc1[s].keys())
            groups2 = set(enc2[s].keys())
            assert len(groups1.symmetric_difference(groups2)) == 0, "Input has different groups"
            for g in groups1:
                outputs[s][g] = alpha * enc1[s][g] + (1 - alpha) * enc2[s][g]

        return outputs


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
