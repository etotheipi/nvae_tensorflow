import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa


class SqueezeExciteLayer(L.Layer): def __init__(self, ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        
    def build(self, input_shape):
        orig_size = input_shape[-1]
        squeeze_size = max(orig_size // self.ratio, 4)
        
        self.pool = L.GlobalAveragePooling2D()
        self.dense1 = L.Dense(num_hidden, activation='relu')
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
                 filters,
                 kernel_size,
                 downsample=False,
                 upsample=False,
                 depthwise=False,
                 bias=True,
                 weight_norm=True,
                 spectral_norm=True,
                 dilation=1,
                 activation='linear',
                 *args,
                 **kwargs):
        """
        A wrapper around tf.keras.layers.Conv2D to streamline inclusion of common ops, primarily
        weight/spectral normalization, and choosing depthwise convolutions.
        """
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.upsample = upsample
        self.depthwise = depthwise
        self.bias = bias
        self.weight_norm = weight_norm
        self.spectral_norm = spectral_norm
        self.dilation = dilation
        self.activation = activation

        assert not (downsample and upsample), 'Cannot upsample and downsample simultaneously'
        # NVLabs implementation uses -1 to indicate (2,2) upsampling.
        self.upsample_layer = None
        if upsample:
            self.upsample_layer = L.UpSampling2D((2,2), interpolation='nearest')

        self.conv = L.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1 if not downsample else 2,
            groups=1 if not depthwise else filters,
            bias=bias,
            dilation=dilation,
            activation=activation,
            padding='same')

        if weight_norm:
            self.conv = tfa.layers.WeightNormalization(self.conv)

        if spectral_norm:
            self.conv = tfa.layers.SpectralNormalization(self.conv)

    def call(self, x, training=False):
        if self.updownsample == 'up':
            x = self.upsample_layer(x)

        return self.conv(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "depthwise": self.depthwise,
            "bias": self.bias,
            "weight_norm": self.weight_norm,
            "spectral_norm": self.spectral_norm,
            "dilation": self.dilation,
            "activation": self.activation,
        })


class ResidualDecoderCell(L.Layer):
    def __init__(self,
            strides=1,
            expand_ratio=6,
            se_ratio=8,
            bn_momentum=0.95,
            gamma_reg=None,
            w_bias=True,
            w_upsample=False,
            res_scalar=0.1,
            *args,
            **kwargs):
        
        super().__init__(*args, **kwargs)
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.gamma_reg = gamma_reg
        self.w_bias = w_bias
        self.w_upsample = w_upsample
        self.res_scalar = res_scalar

    def build(self, input_shape):
        # Num channels, and num expanded channels
        # TODO: All these Conv2Ds need spectral-normalization and weight-normalization!
        num_c = input_shape[-1]
        num_ec = num_c * self.expand_ratio

        self.bn_layers = [L.BatchNormalization(
            momentum=self.bn_momentum,
            gamma_regularizer=self.gamma_reg) for _ in range(4)]

        self.conv_expand = NvaeConv2D(filters=num_ec, kernel_size=(1, 1))

        # Depthwise separable convolution, with possible upsample (if strides=-1)
        self.conv_depth1 = NvaeConv2D(filters=num_ec, kernel_size=(5, 5), strides=self.strides, depthwise=True)
        self.conv_depth2 = NvaeConv2D(filters=num_ec, kernel_size=(1, 1))

        self.conv_reduce = NvaeConv2D(filters=num_c, kernel_size=(1, 1), bias=False)
        self.se_layer = SqueezeExciteLayer(self.se_ratio)

    def call(self, batch_input, training=None):
        x = batch_input

        x = self.bn_layers[0](x, training=training)
        x = self.conv_expand(x)

        x = self.bn_layers[1](x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_depth1(x)
        x = self.conv_depth2(x)

        x = self.bn_layers[2](x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_reduce(x)

        x = self.bn_layers[3](x, training=training)
        x = self.se_layer(x)

        output = L.Add()([batch_input, self.res_scalar * x])
        
        if self.upsample_layer is not None:
            output = self.upsample_layer(output)
            
        return output

    def create_model(self, input_shape_3d):
        # This is really just so I can see a summary and use plot_model
        x = inputs = L.Input(shape=input_shape_3d)

        x = self.bn_layers[0](x)
        x = self.conv_expand(x)

        x = self.bn_layers[1](x)
        x = tf.keras.activations.swish(x)
        x = self.conv_depth1(x)
        x = self.conv_depth2(x)

        x = self.bn_layers[2](x)
        x = tf.keras.activations.swish(x)
        x = self.conv_reduce(x)

        x = self.bn_layers[3](x)
        x = self.se_layer(x)
        
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)

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
            strides=1,
            se_ratio=8,
            bn_momentum=0.95,
            gamma_reg=None,
            w_bias=True,
            w_downsample=False,
            res_scalar=0.1,
            *args,
            **kwargs):
        
        super().__init__(*args, **kwargs)
        self.strides = strides
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.gamma_reg = gamma_reg
        self.w_bias = w_bias
        self.w_downsample = w_downsample
        self.res_scalar = res_scalar

    def build(self, input_shape):
        num_c = input_shape[-1]

        self.bn_layers = [L.BatchNormalization(
            momentum=self.bn_momentum,
            gamma_regularizer=self.gamma_reg) for _ in range(2)]

        # If this is a downsample cell, only the first conv gets the strides=2 arg
        strides = [2, 1] if self.w_downsample else [1, 1]
        self.conv_3x3s = [NvaeConv2D(filters=num_c, kernel_size=(3, 3), strides=strides[i]) for i in range(2)]
        self.se_layer = SqueezeExciteLayer(self.se_ratio)
        self.downsampler = FactorizedDownsample(num_c * 2) if self.w_downsample else None


    def call(self, batch_input, training=None):
        x = batch_input

        x = self.bn_layers[0](x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s[0](x)

        x = self.bn_layers[1](x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s[1](x)

        x = self.se_layer(x)

        output = L.Add()([batch_input, self.res_scalar * x])
        
        if self.downsampler is not None:
            output = self.downsampler(output)
        
        return output

    def create_model(self, input_shape_3d):
        # This is really just so I can see a summary and use plot_model
        x = inputs = L.Input(input_shape_3d)

        x = self.bn_layers[0](x)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s[0](x)

        x = self.bn_layers[1](x)
        x = tf.keras.activations.swish(x)
        x = self.conv_3x3s[1](x)

        x = self.se_layer(x)
        
        if self.downsampler is not None:
            x = self.downsampler(x)

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
    def __init__(self, channels_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels_out = channels_out

        quarter = channels_out // 4
        lastqrt = channels_out - 3 * quarter
        self.conv1 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv2 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv3 = L.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
        self.conv4 = L.Conv2D(filters=lastqrt, kernel_size=(1, 1), strides=(2, 2), padding='same')

    def call(self, batch_input):
        stack1 = self.conv1(batch_input[:, :, :, :])
        stack2 = self.conv2(batch_input[:, 1:, :, :])
        stack3 = self.conv3(batch_input[:, :, 1:, :])
        stack4 = self.conv4(batch_input[:, 1:, 1:, :])

        return tf.stack([stack1, stack2, stack3, stack4])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'channels_out': self.channels_out})
        return cfg


class KLDivergence:
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

    @staticmethod
    def vs_unit_normal(self, mu, log_var):
        return -0.5 * tf.reduce_mean(log_var - tf.square(mu) - tf.exp(log_var) + 1)


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




