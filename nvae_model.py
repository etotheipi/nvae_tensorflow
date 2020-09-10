import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa

from nvae_layers import ResidualEncoderCell, ResidualDecoderCell, Sampling, CombinerCell, NvaeConv2D

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


def create_nvae(
    input_shape,
    base_nfilt,
    nscales,
    ngroups,
    nlatent,
    num_preproc_blocks=2,
    num_preproc_cells=2,
    num_postproc_blocks=2,
    num_postproc_cells=2):

    # 
    orig_input_shape = input_shape[-3:]  # initial image shape (W, H, Ch)
    side = input_shape[-2]  # initial image shape (W)
    chan = input_shape[-1]  # initial num channels (Ch)

    combine_samplers = []
    combiner_decs = []

    x = inputs = L.Input(input_shape)

    #####
    # Stem -- Expand from 3 channels to num_channels_enc
    x = NvaeConv2D(base_nfilt, kernel_size=(3, 3))

    #####
    # Preprocessing -- No combiner/latent outputs
    for i_pre in range(num_preproc_blocks):
        
        for i_cell in range(num_preproc_cells):
            last_cell_in_block = (i_cell == num_preproc_cells -1)
            x = ResidualEncoderCell(downsample=last_cell_in_block)(x)
            
        chan = chan * 2
        side = side // 2

    #####
    # Encoder Tower -- aggregate combine_samplers after each group
    for s in range(nscales):
        last_scale = (s == nscales - 1)
        
        for g in range(ngroups):
            last_group = (g == ngroups - 1)
            
            for c in range(ncells):
                last_cell = (c == ncells - 1)
                
                x = ResidualEncoderCell()(x)
                
                if last_cell:
                    #This will contain [left, CombinerCell, right]
                    combine_samplers.append([x, CombinerCell(sampling_shape=(side, side, chan)), None])
                    
            if last_group and not last_scale:
                x = ResidualEncoderCell(downsample=True)(x)
                chan = chan * 2
                side = side // 2
    

    #####
    # Encoder0 -- named this way in the NVLabs code
    x = tf.keras.activations.ELU(x) 
    x = NvaeConv2D(filters=chan, kernel_size=(1, 1))
    x = tf.keras.activations.ELU(x) 
    
    # Can we use variables within functional models?  Well what we want is just a 
    ftr0 = tf.Variable(shape=(side, side, chan))
    
    #####
    # Decoder Tower -- aggregate combine_samplers after each group
    for s in range(nscales):
        last_scale = (s == nscales - 1)
        
        for g in range(ngroups):
            last_group = (g == ngroups - 1)
            
            for c in range(ncells):
                last_cell = (c == ncells - 1)
                
                x = ResidualEncoderCell()(x)
                
                if last_cell:
                    #This will contain [left, CombinerCell, right]
                    combine_samplers.append([x, CombinerCell(sampling_shape=(side, side, chan)), None])
                    
            if last_group and not last_scale:
                x = ResidualEncoderCell(downsample=True)(x)
                chan = chan * 2
                side = side // 2
    
    
    
    model = tf.keras.Model(inputs=inputs, outputs=inputs)
    model.ftr0 = ftr0
    
    return model
    
    
    