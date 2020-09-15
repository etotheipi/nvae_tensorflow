import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa


class SqueezeExciteLayer(L.Layer):
    def __init__(self, ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.pool = None
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):
        orig_size = input_shape[-1]
        squeeze_size = max(orig_size // self.ratio, 4)
        
        self.pool = L.GlobalAveragePooling2D()
        self.dense1 = L.Dense(squeeze_size, activation='relu')
        self.dense2 = L.Dense(orig_size, activation='sigmoid')
        
    def call(self, batch_input):
        x = self.pool(batch_input)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, shape=(-1, 1, 1, batch_input.shape[-1]))
        return x * batch_input
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'ratio': self.ratio})
        return cfg

    
class NvaeConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size,
                 scale_channels=1,
                 downsample=False,
                 upsample=False,
                 depthwise=False,
                 use_bias=True,
                 weight_norm=False,
                 spectral_norm=True,
                 dilation_rate=1,
                 activation='linear',
                 padding='same',
                 abs_channels=None,
                 *args,
                 **kwargs):
        """
        A wrapper around tf.keras.layers.Conv2D to streamline inclusion of common ops, primarily
        weight/spectral normalization, and choosing depthwise convolutions.  Also designed to 
        accommodate chaining Conv2D layers without specifying number of filters or image sizes,
        instead declaring channel-scaling factors relative to inputs and simply indicating
        upsampling or downsampling image size.
        
        In most cases, we will be using one of the following constructs:
        
        Output == input shape:  NvaeConv2D(kernel_size=3)
        Downsample w/ chan*2:   NvaeConv2D(kernel_size=3, scale_channels=2, downsample=True)
        Upsample w/ chan//2:    NvaeConv2D(kernel_size=3, scale_channels=-2, upsample=True)
        """
        super().__init__(*args, **kwargs)

        self.abs_channels = abs_channels
        self.scale_channels = scale_channels
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.upsample = upsample
        self.depthwise = depthwise
        self.use_bias = use_bias
        self.weight_norm = weight_norm
        self.spectral_norm = spectral_norm
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.padding = padding
        self.channels_in = None

        self.conv = None
        self.conv_depth1x1 = None

        assert not (downsample and upsample), 'Cannot upsample and downsample simultaneously'
        assert [self.scale_channels, self.abs_channels] != [None, None], "Specify at either scale_channels, or abs_filt"
        # NVLabs implementation uses -1 to indicate (2,2) upsampling.
        self.upsample_layer = None
        if upsample:
            self.upsample_layer = L.UpSampling2D((2,2), interpolation='nearest')

    def build(self, input_shape):
        self.channels_in = input_shape[-1]
        if self.abs_channels is None:
            assert self.scale_channels != 0
            if self.scale_channels > 0:
                self.abs_channels = self.channels_in * self.scale_channels
            else:
                assert self.channels_in % abs(self.scale_channels) == 0, "Input channels not a multiple of scalar"
                self.abs_channels = self.channels_in // abs(self.scale_channels)

        self.conv = L.Conv2D(
            filters=self.abs_channels,
            kernel_size=self.kernel_size,
            strides=1 if not self.downsample else 2,
            groups=1 if not self.depthwise else self.abs_channels,
            use_bias=self.use_bias,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            padding=self.padding)

        self.conv_depth1x1 = None
        if self.depthwise:
            self.conv_depth1x1 = L.Conv2D(self.abs_channels, kernel_size=(1, 1))

        if self.weight_norm:
            self.conv = tfa.layers.WeightNormalization(self.conv)
            if self.weight_norm and self.conv_depth1x1:
                self.conv_depth1x1 = tfa.layers.WeightNormalization(self.conv_depth1x1)

        if self.spectral_norm:
            self.conv = tfa.layers.SpectralNormalization(self.conv)
            if self.depthwise and self.conv_depth1x1:
                self.conv_depth1x1 = tfa.layers.SpectralNormalization(self.conv_depth1x1)
        else:
            print('Spectral Norm is disabled!')

    def call(self, x, training=False):
        if self.upsample:
            x = self.upsample_layer(x)

        x = self.conv(x, training=training)

        if self.depthwise:
            x = self.conv_depth1x1(x, training=training)

        return x
       
    def get_config(self):
        # TODO: UPDATE this
        cfg = super().get_config()
        cfg.update({
            "abs_channels": self.abs_channels,
            "scale_channels": self.scale_channels,
            "kernel_size": self.kernel_size,
            "depthwise": self.depthwise,
            "use_bias": self.use_bias,
            "weight_norm": self.weight_norm,
            "spectral_norm": self.spectral_norm,
            "dilation_rate": self.dilation_rate,
            "activation": self.activation,
        })


class ResidualDecoderCell(L.Layer):
    def __init__(self,
                 upsample=False,
                 expand_ratio=6,
                 se_ratio=16,
                 bn_momentum=0.95,
                 gamma_reg=None,
                 use_bias=True,
                 res_scalar=0.1,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.gamma_reg = gamma_reg
        self.use_bias = use_bias
        self.upsample = upsample
        self.res_scalar = res_scalar
        
        self.conv_depthw = None
        self.conv_expand = None
        self.conv_reduce = None
        self.upsample_residual = None
        self.upsample_conv1x1 = None

    def build(self, input_shape):
        #print('ResDecCell build shape:', input_shape)
        # Num channels, and num expanded channels
        # TODO: All these Conv2Ds need spectral-normalization and weight-normalization!
        num_c = input_shape[-1]
        num_ec = num_c * self.expand_ratio

        self.bn_layers = [L.BatchNormalization(
            momentum=self.bn_momentum,
            gamma_regularizer=self.gamma_reg) for _ in range(4)]

        self.conv_expand = NvaeConv2D(kernel_size=(1, 1), scale_channels=self.expand_ratio)

        # Depthwise separable convolution, with possible upsample
        if self.upsample:
            self.conv_depthw = NvaeConv2D(kernel_size=(5, 5),
                                          depthwise=True,
                                          upsample=True,
                                          scale_channels=-2,
                                          use_bias=False,
                                          weight_norm=False)
            self.upsample_residual = NvaeConv2D(kernel_size=(1,1), upsample=True, scale_channels=-2)
        else:
            self.conv_depthw = NvaeConv2D(kernel_size=(5, 5),
                                          depthwise=True,
                                          use_bias=False,
                                          weight_norm=False)
            
        self.conv_reduce = NvaeConv2D(kernel_size=(1, 1),
                                      scale_channels=-self.expand_ratio,
                                      use_bias=False,
                                      weight_norm=False)

        self.se_layer = SqueezeExciteLayer(self.se_ratio)

    def call(self, batch_input, training=None):
        x = batch_input

        x = self.bn_layers[0](x, training=training)
        x = self.conv_expand(x, training=training)

        x = self.bn_layers[1](x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_depthw(x, training=training)

        x = self.bn_layers[2](x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_reduce(x, training=training)

        x = self.bn_layers[3](x, training=training)
        x = self.se_layer(x, training=training)

        residual = batch_input
        if self.upsample:
            residual = self.upsample_residual(residual, training=training)
            
        output = L.Add()([residual, self.res_scalar * x])
        
        
        return output

    def create_model(self, input_shape_3d):
        # This is really just so I can see a summary and use plot_model
        x = inputs = L.Input(shape=input_shape_3d)

        x = self.bn_layers[0](x)
        x = self.conv_expand(x)

        x = self.bn_layers[1](x)
        x = tf.keras.activations.swish(x)
        x = self.conv_depthw(x)
        x = self.conv_depth2(x)

        x = self.bn_layers[2](x)
        x = tf.keras.activations.swish(x)
        x = self.conv_reduce(x)

        x = self.bn_layers[3](x)
        x = self.se_layer(x)
        
        residual = inputs
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)
            residual = self.upsample_layer(residual)

        output = L.Add()([inputs, self.res_scalar*x])

        return tf.keras.Model(inputs=inputs, outputs=output)

    def get_config(self, *args, **kwargs):
        cfg = super().get_config()
        cfg.update({
            'expand_ratio', self.expand_ratio,
            'se_ratio', self.se_ratio,
            'bn_momentum', self.bn_momentum,
            'gamma_reg', self.gamma_reg
        })
        return cfg


class ResidualEncoderCell(L.Layer):
    def __init__(self,
                 downsample=False,
                 se_ratio=16,
                 bn_momentum=0.95,
                 gamma_reg=None,
                 use_bias=True,
                 res_scalar=0.1,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.gamma_reg = gamma_reg
        self.use_bias = use_bias
        self.downsample = downsample
        self.res_scalar = res_scalar

        self.bn0 = None
        self.bn1 = None
        self.conv_3x3s_0 = None
        self.conv_3x3s_1 = None
        self.se_layer = None
        self.downsample_layer = None

    def build(self, input_shape):
        #print('ResEncCell build shape:', input_shape)

        self.bn0 = L.BatchNormalization(momentum=self.bn_momentum, gamma_regularizer=self.gamma_reg)
        self.bn1 = L.BatchNormalization(momentum=self.bn_momentum, gamma_regularizer=self.gamma_reg)

        if self.downsample:
            self.conv_3x3s_0 = NvaeConv2D(kernel_size=(3, 3), downsample=True, scale_channels=2)
            self.conv_3x3s_1 = NvaeConv2D(kernel_size=(3, 3))
        else:
            self.conv_3x3s_0 = NvaeConv2D(kernel_size=(3, 3))
            self.conv_3x3s_1 = NvaeConv2D(kernel_size=(3, 3))

        self.se_layer = SqueezeExciteLayer(self.se_ratio)
        
        if self.downsample:
            self.downsample_layer = FactorizedDownsample()

    def call(self, batch_input, training=None):
        x = batch_input

        x = self.bn0(x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s_0(x, training=training)

        x = self.bn1(x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s_1(x, training=training)

        x = self.se_layer(x)

        residual = batch_input
        if self.downsample:
            residual = self.downsample_layer(residual, training=training)
            
        output = L.Add()([residual, self.res_scalar * x])
        
        return output

    def create_model(self, input_shape_3d):
        # This is really just so I can see a summary and use plot_model
        x = inputs = L.Input(input_shape_3d)
        
        x = self.bn0(x)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s_0(x)

        x = self.bn1(x)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s_1(x)

        x = self.se_layer(x)

        output = L.Add()([inputs, self.res_scalar*x])

        return tf.keras.Model(inputs=inputs, outputs=output)

    def get_config(self, *args, **kwargs):
        cfg = super().get_config()
        cfg.update({
            'se_ratio', self.se_ratio,
            'bn_momentum', self.bn_momentum,
            'gamma_reg', self.gamma_reg
        })
        return cfg


class FactorizedDownsample(L.Layer):
    """
    This is used in the NVLabs implementation for the skip/residual connections during down-scaling
    """
    def __init__(self, channels_out=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels_out = channels_out

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None

    def build(self, input_shape):
        #print('FactDown build shape:', input_shape)
        channels_in = input_shape[-1]
        if self.channels_out is None:
            self.channels_out = channels_in * 2

        quarter = self.channels_out // 4
        lastqrt = self.channels_out - 3 * quarter
        self.conv1 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv2 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv3 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv4 = L.Conv2D(filters=lastqrt, kernel_size=(1, 1), strides=(2, 2), padding='same')

    def call(self, batch_input, training=False):
        stack1 = self.conv1(batch_input[:, :, :, :], training=training)
        stack2 = self.conv2(batch_input[:, 1:, :, :], training=training)
        stack3 = self.conv3(batch_input[:, :, 1:, :], training=training)
        stack4 = self.conv4(batch_input[:, 1:, 1:, :], training=training)

        out = L.Concatenate(axis=-1)([stack1, stack2, stack3, stack4])
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'channels_out': self.channels_out})
        return cfg


class KLDivergence:
    @staticmethod
    def vs_unit_normal(mu, logvar):
        return -0.5 * tf.reduce_mean(logvar - tf.square(mu) - tf.exp(logvar) + 1)
    
    # This would be useful for the residual-normal construct
    @staticmethod
    def two_guassians(mu1, sigma_sq_1, mu2, sigma_sq_2):
        # https://stats.stackexchange.com/a/7449
        term1 = 0.5 * tf.math.log(sigma_sq_2 / sigma_sq_1)
        term2 = (sigma_sq_1 + (mu1 - mu2) ** 2) / (2 * sigma_sq_2)
        return tf.reduce_mean(term1 + term2 - 0.5)

    @staticmethod
    def two_guassians_logvar(mu1, logvar_1, mu2, logvar_2):
        term1 = 0.5 * (logvar_2 - logvar_1)
        term2 = (tf.exp(logvar_1) + (mu1 - mu2) ** 2) / (2 * tf.exp(logvar_2))
        return tf.reduce_mean(term1 + term2 - 0.5)


class Sampling(L.Layer):
    """
    Uses (z_mean, z_logvar) to sample z, the vector encoding a digit.
    """
    def __init__(self, loss_scalar=1.0, auto_loss=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_scalar = loss_scalar
        self.sample_out_shape = None
        self.auto_loss = auto_loss  # if True, use self.add_loss(kl_term) in call()

    def build(self, input_shape):
        #print('Sampling build shape:', input_shape)
        if isinstance(input_shape, (tuple, list)):
            shape1, shape2 = input_shape
            assert list(shape1) == list(shape2), f'Sample input mismatch: {shape1} != {shape2}'
        else:
            # One tensor passed in, needs to be split into two:
            shape1 = input_shape[:-1] + (input_shape[-1] // 2,)

        self.sample_out_shape = shape1
        
    def call(self, inputs, training=False):
        if isinstance(inputs, (tuple, list)):
            z_mean, z_logvar = inputs
        else:
            # If it's a single tensor, assume half channels are mu, half are log-var
            z_mean = inputs[:, :, :, self.sample_out_shape[-1]:]
            z_logvar = inputs[:, :, :, :self.sample_out_shape[-1]]

        if self.auto_loss:
            kl_loss = KLDivergence.vs_unit_normal(z_mean, z_logvar)
            self.add_loss(self.loss_scalar * kl_loss)

        if not training:
            return z_mean

        epsilon = tf.random.normal(shape=tf.shape(z_logvar))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

    def compute_output_shape(self, input_shape):
        return self.sample_out_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'loss_scalar': self.loss_scalar
        })
        return cfg



class MergeCellPeak(L.Layer):
    """
    The peak merge cell is a bit different than the rest, taking the trainable
    param at the peak and the z0 sample as inputs.  
    """
    def __init__(self, num_latent, kl_loss_scalar=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_latent = num_latent
        self.kl_loss_scalar = kl_loss_scalar
        self.orig_shape = None
        self.encoder0 = None
        self.conv_enc_1 = None
        self.sampling_enc = None
        self.conv_dec_0 = None
        self.generate = False

    def build(self, input_shape):
        left_shape, right_shape = input_shape
        assert list(left_shape) == list(right_shape), f'Left={left_shape} != Right={right_shape}'
        self.orig_shape = right_shape
        self.encoder0 = NvaeConv2D(kernel_size=(1, 1))
        self.conv_enc_1 = NvaeConv2D(kernel_size=(3, 3), abs_channels=2*self.num_latent)
        self.sampling_enc = Sampling(auto_loss=False)
        self.conv_dec_0 = NvaeConv2D(kernel_size=(1, 1), abs_channels=self.orig_shape[-1])

    def set_generate_mode(self, generative=True):
        self.generate = generative

    def call(self, input_pair, training=False):
        if self.generate:
            assert not training, 'Cannot generate in training mode!'
            _, ftr0 = input_pair
            side = self.orig_shape[-3]
            z = tf.random.normal((side, side, self.num_latent))
            # Need to set these for loss calc later, even though not in training mode
            mu_q = tf.zeros_like(z)
            logvar_q = tf.zeros_like(z)
        else:
            # With encoder
            s_enc, ftr0 = input_pair

            s_enc = tf.keras.activations.elu(s_enc)
            s_enc = self.encoder0(s_enc, training=training)
            s_enc = tf.keras.activations.elu(s_enc)

            params_q = self.conv_enc_1(s_enc, training=training)
            mu_q = params_q[:, :, :, self.num_latent:]
            logvar_q = params_q[:, :, :, :self.num_latent]
            z = self.sampling_enc([mu_q, logvar_q], training=training)
            
            batch_size = tf.shape(s_enc)[0]
            ftr0 = tf.tile(ftr0, [batch_size, 1, 1, 1])

        merge = L.Concatenate(axis=-1)([z, ftr0])
        s_out = self.conv_dec_0(merge, training=training)

        # KL-divergence of q(z|x) against N(0,1)
        kl_term = KLDivergence.vs_unit_normal(mu_q, logvar_q)
        self.add_loss(self.kl_loss_scalar * kl_term)

        return s_out


class MergeCell(L.Layer):
    """
    We initialize with the output of the encoder side, which is created before
    """

    def __init__(self, num_latent, kl_loss_scalar=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_latent = num_latent
        self.kl_loss_scalar = kl_loss_scalar
        self.orig_chan = None
        self.conv_enc_0 = None
        self.conv_enc_1 = None
        self.sampling_enc = None
        self.conv_dec_0 = None
        self.conv_dec_1 = None
        self.sampling_dec = None
        self.generate = False

    def build(self, input_shape):
        s_enc_shape, s_dec_shape = input_shape
        self.orig_chan = s_enc_shape[-1]

        # print('CombSample build shape:', input_shape)
        assert list(s_enc_shape) == list(s_dec_shape), f's_enc:{s_enc_shape} != s_dec:{s_dec_shape}'
        self.conv_enc_0 = NvaeConv2D(kernel_size=(1, 1))
        self.conv_enc_1 = NvaeConv2D(kernel_size=(3, 3), abs_channels=2*self.num_latent)

        self.conv_dec_0 = NvaeConv2D(kernel_size=(1, 1), abs_channels=self.orig_chan)
        self.conv_dec_1 = NvaeConv2D(kernel_size=(3, 3), abs_channels=2*self.num_latent)

        self.sampling_enc = Sampling(auto_loss=False)
        self.sampling_dec = Sampling(auto_loss=False)

    def set_generate_mode(self, generative=True):
        self.generate = generative

    def call(self, s_enc_dec_pair, training=False):
        if self.generate:
            assert not training, 'Cannot generate in training mode!'
            _, s_dec = s_enc_dec_pair
            print('Generate mode')
            x = tf.keras.activations.elu(s_dec)
            params_p = self.conv_dec_1(x, training=training)
            mu_p = params_p[:, :, :, self.num_latent:]
            logvar_p = params_p[:, :, :, :self.num_latent]
            z = self.sampling_dec([mu_p, logvar_p], training=training)
            s_out = self.conv_dec_0(L.Concatenate(axis=-1)([z, s_dec]), training=training)
        else:
            # With encoder
            s_enc, s_dec = s_enc_dec_pair
            ftr = s_enc + self.conv_enc_0(s_dec, training=training)
            params_q = self.conv_enc_1(ftr, training=training)
            mu_q = params_q[:, :, :, self.num_latent:]
            logvar_q = params_q[:, :, :, :self.num_latent]

            z = self.sampling_enc([mu_q, logvar_q], training=training)
            s_out = self.conv_dec_0(L.Concatenate(axis=-1)([z, s_dec]), training=training)

            # Before returning, we need to compute the DecoderSampler and KL terms
            s_dec = tf.keras.activations.elu(s_dec)
            params_p = self.conv_dec_1(s_dec, training=training)
            mu_p = params_p[:, :, :, self.num_latent:]
            logvar_p = params_p[:, :, :, :self.num_latent]

            # KL-divergence of q(z|x) against p(z|x) estimate
            kl_term = KLDivergence.two_guassians_logvar(mu_q, logvar_q, mu_p, logvar_p)
            self.add_loss(self.kl_loss_scalar * kl_term)

        return s_out


