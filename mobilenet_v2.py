# Created by github/Gustavz
# MobileNetV2 Implementation in Keras
# Paper: https://arxiv.org/abs/1801.04381

# python 2 compability
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
import keras.layers as KL
import keras.models as KM
from keras.utils.vis_utils import plot_model


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)


def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1),block_id=1, train_bn=False):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Id of the conv_block
        train_bn: Boolean. Train or freeze Batch Norm layers
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = KL.Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv{}'.format(block_id))(inputs)
    x = BatchNorm(axis=channel_axis, name='conv{}_bn'.format(block_id))(x, training = train_bn)
    return KL.Activation(relu6, name='conv{}_relu'.format(block_id))(x)


def bottleneck(inputs, filters, kernel, t, s, r=False, alpha=1.0, block_id=1, train_bn = False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
        block_id: Id of the bottleneck
        train_bn: Boolean. Train or freeze Batch Norm layers
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    filters = int(alpha * filters)

    x = conv_block(inputs, tchannel, alpha, (1, 1), (1, 1),block_id=block_id,train_bn=train_bn)

    x = KL.DepthwiseConv2D(kernel,
                    strides=(s, s),
                    depth_multiplier=1,
                    padding='same',
                    name='conv_dw_{}'.format(block_id))(x)
    x = BatchNorm(axis=channel_axis,name='conv_dw_{}_bn'.format(block_id))(x, training=train_bn)
    x = KL.Activation(relu6, name='conv_dw_{}_relu'.format(block_id))(x)

    x = KL.Conv2D(filters,
                    (1, 1),
                    strides=(1, 1),
                    padding='same',
                    name='conv_pw_{}'.format(block_id))(x)
    x = BatchNorm(axis=channel_axis, name='conv_pw_{}_bn'.format(block_id))(x, training=train_bn)

    if r:
        x = KL.add([x, inputs], name='res{}'.format(block_id))
    return x


def inverted_residual_block(inputs, filters, kernel, t, strides, n, alpha, block_id, train_bn=False):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
        block_id: Id of the inv_res_block, increments itself if layers repeat
        train_bn: Boolean. Train or freeze Batch Norm layers
    # Returns
        Output tensor.
    """

    x = bottleneck(inputs, filters, kernel, t, strides, False, alpha, block_id, train_bn)

    for i in range(1, n):
        block_id += 1
        x = bottleneck(x, filters, kernel, t, 1, True, alpha, block_id, train_bn)

    return x


def mobilenetv2 (inputs, k, alpha = 1.0, train_bn = False):
    """MobileNetv2
    This function defines a MobileNetv2 model.
    # Arguments
        inputs: Inuput Tensor, e.g. an image
        k: Number of classes
        alpha: Width Multiplier
        train_bn: Boolean. Train or freeze Batch Norm layres
    # Returns
        model
    """

    x = conv_block(inputs, 32, alpha, (3, 3), strides=(2, 2), block_id=0, train_bn=train_bn)                    # Input Res: 1

    x = inverted_residual_block(x, 16,  (3, 3), t=1, strides=1, n=1, alpha=1.0, block_id=1, train_bn=train_bn)	# Input Res: 1/2
    x = inverted_residual_block(x, 24,  (3, 3), t=6, strides=2, n=2, alpha=1.0, block_id=2, train_bn=train_bn)	# Input Res: 1/2
    x = inverted_residual_block(x, 32,  (3, 3), t=6, strides=2, n=3, alpha=1.0, block_id=4, train_bn=train_bn)	# Input Res: 1/4
    x = inverted_residual_block(x, 64,  (3, 3), t=6, strides=2, n=4, alpha=1.0, block_id=7, train_bn=train_bn)	# Input Res: 1/8
    x = inverted_residual_block(x, 96,  (3, 3), t=6, strides=1, n=3, alpha=1.0, block_id=11, train_bn=train_bn)	# Input Res: 1/8
    x = inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3, alpha=1.0, block_id=14, train_bn=train_bn)	# Input Res: 1/16
    x = inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=17, train_bn=train_bn)	# Input Res: 1/32

    x = conv_block(x, 1280, alpha, (1, 1), strides=(1, 1), block_id=18, train_bn=train_bn)                      # Input Res: 1/32

    x = KL.GlobalAveragePooling2D()(x)
    x = KL.Reshape((1, 1, 1280))(x)
    x = KL.Dropout(0.3, name='Dropout')(x)
    x = KL.Conv2D(k, (1, 1), padding='same')(x)

    x = KL.Activation('softmax', name='softmax')(x)
    output = KL.Reshape((k,))(x)

    model = KM.Model(inputs, output)
    plot_model(model, to_file='MobileNetv2.png', show_shapes=True)

    return model
