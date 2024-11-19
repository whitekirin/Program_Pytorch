# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# SENet
    # block = layers.GlobalAveragePooling2D()(residual)
    # block = layers.Dense(units = residual.shape[-1] // 16, activation = "relu")(block)
    # block = layers.Dense(units = residual.shape[-1], activation = "sigmoid")(block)
    # block = Reshape((1, 1, residual.shape[-1]))(block)
    # residual = Multiply()([residual, block])



from keras import backend
from keras import layers
from keras.layers import Reshape, Multiply, Conv1D
import math

def Xception_indepentment(input_shape=None):

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    img_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(
        32, (3, 3), strides=(2, 2), use_bias=False, name="block1_conv1"
    )(img_input)
    x = layers.BatchNormalization(axis=channel_axis, name="block1_conv1_bn")(x)
    x = layers.Activation("relu", name="block1_conv1_act")(x
                                                           )
    x = layers.Conv2D(64, (3, 3), use_bias=False, name="block1_conv2")(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block1_conv2_bn")(x)
    x = layers.Activation("relu", name="block1_conv2_act")(x)

    residual = layers.Conv2D(
        128, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    # 注意力機制區域
    kernel = int(abs((math.log(residual.shape[-1], 2) +  1) / 2))
    if kernel % 2:
        kernel_size = kernel
    else:
        kernel_size = kernel + 1
    
    block = layers.GlobalAveragePooling2D()(residual)
    block = Reshape(target_shape = (residual.shape[-1], 1))(block)
    block = Conv1D(filters = 1, kernel_size = kernel_size, padding = "same", use_bias = False, activation = "sigmoid")(block)
    block = Reshape((1, 1, residual.shape[-1]))(block)
    residual = Multiply()([residual, block])

    x = layers.SeparableConv2D(
        128, (3, 3), padding="same", use_bias=False, name="block2_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block2_sepconv1_bn")(
        x
    )
    x = layers.Activation("relu", name="block2_sepconv2_act")(x)

    x = layers.SeparableConv2D(
        128, (3, 3), padding="same", use_bias=False, name="block2_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block2_sepconv2_bn")(
        x
    )

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block2_pool"
    )(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        256, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    # 注意力機制區域
    kernel = int(abs((math.log(residual.shape[-1], 2) +  1) / 2))
    if kernel % 2:
        kernel_size = kernel
    else:
        kernel_size = kernel + 1
    
    block = layers.GlobalAveragePooling2D()(residual)
    block = Reshape(target_shape = (residual.shape[-1], 1))(block)
    block = Conv1D(filters = 1, kernel_size = kernel_size, padding = "same", use_bias = False, activation = "sigmoid")(block)
    block = Reshape((1, 1, residual.shape[-1]))(block)
    residual = Multiply()([residual, block])

    x = layers.Activation("relu", name="block3_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding="same", use_bias=False, name="block3_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block3_sepconv1_bn")(
        x
    )
    x = layers.Activation("relu", name="block3_sepconv2_act")(x)

    x = layers.SeparableConv2D(
        256, (3, 3), padding="same", use_bias=False, name="block3_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block3_sepconv2_bn")(x)

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block3_pool"
    )(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        728, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    # 注意力機制區域
    kernel = int(abs((math.log(residual.shape[-1], 2) +  1) / 2))
    if kernel % 2:
        kernel_size = kernel
    else:
        kernel_size = kernel + 1
    
    block = layers.GlobalAveragePooling2D()(residual)
    block = Reshape(target_shape = (residual.shape[-1], 1))(block)
    block = Conv1D(filters = 1, kernel_size = kernel_size, padding = "same", use_bias = False, activation = "sigmoid")(block)
    block = Reshape((1, 1, residual.shape[-1]))(block)
    residual = Multiply()([residual, block])

    x = layers.Activation("relu", name="block4_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block4_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block4_sepconv1_bn")(
        x
    )
    x = layers.Activation("relu", name="block4_sepconv2_act")(x)

    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block4_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block4_sepconv2_bn")(
        x
    )

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block4_pool"
    )(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = "block" + str(i + 5)

        x = layers.Activation("relu", name=prefix + "_sepconv1_act")(x)
        x = layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv1",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + "_sepconv1_bn"
        )(x)
        x = layers.Activation("relu", name=prefix + "_sepconv2_act")(x)

        x = layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv2",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + "_sepconv2_bn"
        )(x)
        x = layers.Activation("relu", name=prefix + "_sepconv3_act")(x)

        x = layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv3",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + "_sepconv3_bn"
        )(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(
        1024, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    # 注意力機制區域
    kernel = int(abs((math.log(residual.shape[-1], 2) +  1) / 2))
    if kernel % 2:
        kernel_size = kernel
    else:
        kernel_size = kernel + 1
    
    block = layers.GlobalAveragePooling2D()(residual)
    block = Reshape(target_shape = (residual.shape[-1], 1))(block)
    block = Conv1D(filters = 1, kernel_size = kernel_size, padding = "same", use_bias = False, activation = "sigmoid")(block)
    block = Reshape((1, 1, residual.shape[-1]))(block)
    residual = Multiply()([residual, block])

    x = layers.Activation("relu", name="block13_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block13_sepconv1"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="block13_sepconv1_bn"
    )(x)
    x = layers.Activation("relu", name="block13_sepconv2_act")(x)

    x = layers.SeparableConv2D(
        1024, (3, 3), padding="same", use_bias=False, name="block13_sepconv2"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="block13_sepconv2_bn"
    )(x)

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block13_pool"
    )(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(
        1536, (3, 3), padding="same", use_bias=False, name="block14_sepconv1"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="block14_sepconv1_bn"
    )(x)
    x = layers.Activation("relu", name="block14_sepconv1_act")(x)

    x = layers.SeparableConv2D(
        2048, (3, 3), padding="same", use_bias=False, name="block14_sepconv2"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="block14_sepconv2_bn"
    )(x)
    x = layers.Activation("relu", name="block14_sepconv2_act")(x)

    return img_input, block