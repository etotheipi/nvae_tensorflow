import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa

from nvae_layers import ResidualEncoderCell, ResidualDecoderCell, Sampling

class NVAE(tf.keras.Model):
    def __init__(self, input_shape, base_nfilt, nscales, ngroups, nlatent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_scales = nscales
        self.groups_per_scale = ngroups
        self.latent_per_group = nlatent

        assert input_shape[-3] != input_shape[-2], 'Image shape must be square'
        self.input_shape = input_shape[-3:]  # initial image shape (W, H, Ch)
        self.orig_side = input_shape[-2]  # initial image shape (W)
        self.orig_chan = input_shape[-1]  # initial num channels (Ch)

        self.base_nfilt = base_nfilt

        sm_scale_factor = 2 ** (self.num_scales - 1)
        assert self.orig_side % sm_scale_factor == 0, f'Image size must be multiple of {sm_scale_factor}'
        self.smallest_scale = self.orig_side // sm_scale_factor
        self.largest_num_chan = self.base_nfilt * sm_scale_factor

        # This is the learnable parameter at the peak between encoder & decoder
        self.h_peak = self.add_weight(shape=(self.smallest_scale, self.smallest_scale, self.largest_num_chan))

        self.encoder_blocks = []
        self.decoder_blocks = []
        self.decoder_blocks = []
        
        for s in range()


def create_nvae(input_shape, base_nfilt, nscales, ngroups, nlatent)


