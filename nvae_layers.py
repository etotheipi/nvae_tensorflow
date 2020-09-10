import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa


class SqueezeExciteLayer(L.Layer):
    def __init__(self, ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        
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
                 weight_norm=True,
                 spectral_norm=False,
                 dilation_rate=1,
                 activation='linear',
                 padding='same',
                 abs_channels=None,
                 *args,
                 **kwargs):
        """
        A wrapper around tf.keras.layers.Conv2D to streamline inclusion of common ops, primarily
        weight/spectral normalization, and choosing depthwise convolutions.  Also designed to 
        accommodate chaining Conv2D layers without specifying number of filters, instead declaring
        channel-scaling factors relative to inputs (i.e. 2 for doubling channels, -2 for halving)
        
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
            x = self.conv_depth1x1(x)

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
            se_ratio=8,
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
            self.conv_depthw = NvaeConv2D(kernel_size=(5, 5), depthwise=True, upsample=True, scale_channels=-2)
            self.upsample_residual = NvaeConv2D(kernel_size=(1,1), upsample=True, scale_channels=-2)
        else:
            self.conv_depthw = NvaeConv2D(kernel_size=(5, 5), depthwise=True)
            
        self.conv_reduce = NvaeConv2D(kernel_size=(1, 1), scale_channels=-self.expand_ratio, use_bias=False)
        self.se_layer = SqueezeExciteLayer(self.se_ratio)

    def call(self, batch_input, training=None):
        x = batch_input

        x = self.bn_layers[0](x, training=training)
        x = self.conv_expand(x)

        x = self.bn_layers[1](x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_depthw(x)

        x = self.bn_layers[2](x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_reduce(x)

        x = self.bn_layers[3](x, training=training)
        x = self.se_layer(x)

        residual = batch_input
        if self.upsample:
            residual = self.upsample_residual(residual)
            
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
            se_ratio=8,
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

    def build(self, input_shape):
        num_c = input_shape[-1]

        self.bn_layers = [L.BatchNormalization(
            momentum=self.bn_momentum,
            gamma_regularizer=self.gamma_reg) for _ in range(2)]

        self.bn0 = L.BatchNormalization(momentum=self.bn_momentum, gamma_regularizer=self.gamma_reg)
        self.bn1 = L.BatchNormalization(momentum=self.bn_momentum, gamma_regularizer=self.gamma_reg)

        if self.downsample:
            self.conv_3x3s_0 = NvaeConv2D(kernel_size=(3, 3), downsample=True, scale_channels=2)
        else:
            self.conv_3x3s_0 = NvaeConv2D(kernel_size=(3, 3))
            
        self.conv_3x3s_1 = NvaeConv2D(kernel_size=(3, 3))

        self.se_layer = SqueezeExciteLayer(self.se_ratio)
        
        self.downsampler = None
        if self.downsample:
            self.downsampler = FactorizedDownsample()


    def call(self, batch_input, training=None):
        x = batch_input

        x = self.bn0(x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s_0(x)

        x = self.bn1(x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s_1(x)

        x = self.se_layer(x)

        residual = batch_input
        if self.downsample:
            residual = self.downsampler(residual)
            
        output = L.Add()([residual, self.res_scalar * x])
        
        return output

    def create_model(self, input_shape_3d):
        # This is really just so I can see a summary and use plot_model
        x = inputs = L.Input(input_shape_3d)
        
        x = self.bn0(x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s_0(x)

        x = self.bn1(x, training=training)
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
        
    def build(self, input_shape):
        channels_in = input_shape[-1]
        if self.channels_out is None:
            self.channels_out = channels_in * 2

        quarter = self.channels_out // 4
        lastqrt = self.channels_out - 3 * quarter
        print(channels_in, self.channels_out, quarter, lastqrt)
        self.conv1 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv2 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv3 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv4 = L.Conv2D(filters=lastqrt, kernel_size=(1, 1), strides=(2, 2), padding='same')
        

    def call(self, batch_input):
        stack1 = self.conv1(batch_input[:, :, :, :])
        stack2 = self.conv2(batch_input[:, 1:, :, :])
        stack3 = self.conv3(batch_input[:, :, 1:, :])
        stack4 = self.conv4(batch_input[:, 1:, 1:, :])

        out = L.Concatenate(axis=-1)([stack1, stack2, stack3, stack4])
        print(stack1.shape, stack2.shape, stack3.shape, stack4.shape, out.shape)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'channels_out': self.channels_out})
        return cfg


class KLDivergence:
    @staticmethod
    def vs_unit_normal(self, mu, log_var):
        return -0.5 * tf.reduce_mean(log_var - tf.square(mu) - tf.exp(log_var) + 1)
    
    # This would be useful for the residual-normal construct
    @staticmethod
    def two_guassians(self, mu1, sigma_sq_1, mu2, sigma_sq_2):
        # https://stats.stackexchange.com/a/7449
        term1 = 0.5 * tf.math.log(sigma_sq_2 / sigma_sq_1)
        term2 = (sigma_sq_1 + (mu1 - mu2) ** 2) / (2 * sigma_sq_2)
        return term1 + term2 - 0.5

    @staticmethod
    def two_guassians_log_var(self, mu1, logvar_1, mu2, logvar_2):
        term1 = 0.5 * (logvar_2 - logvar_1)
        term2 = (tf.exp(logvar_1) + (mu1 - mu2) ** 2) / (2 * tf.exp(logvar_2))
        return term1 + term2 - 0.5



class Sampling(L.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """
    def __init__(self, loss_scalar=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_scalar = loss_scalar

    def call(self, inputs, training=False):
        z_mean, z_log_var = inputs

        kl_loss = KLDivergence.vs_unit_normal(z_mean, z_log_var)
        self.add_loss(self.loss_scalar * kl_loss)

        if not training:
            return z_mean

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        print(input_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'loss_scalar': self.loss_scalar
        })
        return cfg



class CombineSampler(L.Layer):
    """
    We initialize with the output of the encoder side, which is created before 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build(self, input_shape):
        self.conv_left = NvaeConv2D(kernel_size=(1, 1), activation='relu')
        self.conv_right = NvaeConv2D(kernel_size=(1, 1), activation='relu')
        self.conv_concat = NvaeConv2D(kernel_size=(1, 1), activation='relu')
        self.sampling = Sampling()
        
    def call(self, enc_dec_pair):
        left, right = enc_dec_pair
        
        left = self.conv_left(left)
        right = self.conv_right(right)
        merge = L.Concatenate(axis=-1)([left, right])
        return self.conv_concat(merge)
        
        
        
