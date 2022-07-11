# -*- coding: utf-8 -*-
# @Time : 2022/7/6 10:59
# @Author : Bao_Zehan
# @File : Pre_Net.py
# @Project : DeepCCP
from tensorflow import keras
import tensorflow as tf

image_mean = tf.constant([131.4479, 123.1393, 80.2495], dtype=tf.float32)
image_std = tf.constant([62.7189, 53.4266, 63.5227], dtype=tf.float32)


def std00(x):
    x = tf.cast(x, tf.float32)
    x = tf.subtract(x, image_mean)
    x = tf.divide(x, image_std)
    return x


@tf.function
def nan_to_zero_tf(x):
    y = tf.where(tf.equal(x, x), x, tf.zeros_like(x))
    return y


class ReluNew(keras.layers.Layer):
    def __init__(self):
        super(ReluNew, self).__init__()
        self.relu1 = keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        relu0 = self.relu1(inputs)
        # convert nan to zero (nan != nan)
        nan_to_zero = nan_to_zero_tf(relu0)
        return nan_to_zero


class InstanceNorm(keras.layers.Layer):
    def __init__(self):
        super(InstanceNorm, self).__init__()

    def call(self, inputs, **kwargs):
        epsilon = 1e-9
        mean, var = tf.nn.moments(inputs, [1, 2], keepdims=True)
        return tf.divide(tf.subtract(inputs, mean), tf.sqrt(tf.add(var, epsilon)))


class Residual(keras.Model):
    def __init__(self, filters, kernel, strides):
        super(Residual, self).__init__()
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.re_conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            padding='SAME'
        )
        self.re_conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            padding='SAME'
        )

    def call(self, inputs, **kwargs):
        conv1 = InstanceNorm()(self.re_conv1(inputs))
        conv2 = ReluNew()(InstanceNorm()(self.re_conv2(ReluNew()(conv1))))
        residual0 = inputs + conv2
        return residual0

    def get_config(self):
        config = super(Residual, self).get_config()
        config.update({"filters": self.filters, "kernel": self.kernel, "strides": self.strides})
        return config


class ResizeConv2d(keras.Model):
    def __init__(self, output_filters, kernel, ):
        super(ResizeConv2d, self).__init__()
        self.output_filters = output_filters
        self.kernel = kernel
        self.up_sample = keras.layers.UpSampling2D(size=2)
        self.rconv1 = keras.layers.Conv2D(
            filters=output_filters,
            kernel_size=kernel,
            strides=1,
            padding='SAME',
            activation='relu'
        )

    def call(self, inputs, **kwargs):
        x_resized = self.up_sample(inputs)
        y = (InstanceNorm()(self.rconv1(x_resized)))
        return y

    def get_config(self):
        config = super(ResizeConv2d, self).get_config()
        config.update({"output_filters": self.output_filters, "kernel": self.kernel})
        return config


class StyleConv(keras.Model):
    def __init__(self, filters):
        super(StyleConv, self).__init__()
        self.conv2d1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding='same'
        )
        self.leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
        self.filters = filters

    def call(self, inputs, **kwargs):
        con1 = self.conv2d1(inputs)
        con1 = self.leaky_relu(con1)
        con1 = InstanceNorm()(con1)
        return con1

    def get_config(self):
        config = super(StyleConv, self).get_config()
        config.update({"filters": self.filters})
        return config


class DoubleConvBN(keras.Model):
    def __init__(self, filters):
        super(DoubleConvBN, self).__init__()

        self.conv2d1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.conv2d2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.relu1 = keras.layers.ReLU()
        self.relu2 = keras.layers.ReLU()
        self.BN1 = keras.layers.BatchNormalization(momentum=0.9)
        self.BN2 = keras.layers.BatchNormalization(momentum=0.9)
        self.filters = filters

    def call(self, inputs, **kwargs):
        conv1 = self.relu1(self.BN1(self.conv2d1(inputs)))
        conv2 = self.relu2(self.BN2(self.conv2d2(conv1)))
        return conv2

    def get_config(self):
        config = super(DoubleConvBN, self).get_config()
        config.update({"filters": self.filters})
        return config


# class Conv2Pool(keras.Model):
#     def __init__(self, filters):
#         super(Conv2Pool, self).__init__()
#         self.filters = filters
#         self.conv_p = keras.layers.Conv2D(
#             filters=filters,
#             kernel_size=3,
#             strides=2,
#             padding='SAME'
#         )
#
#     def call(self, inputs, **kwargs):
#         x = ReluNew()(self.conv_p(inputs))
#         return x
#
#     def get_config(self):
#         config = super(Conv2Pool, self).get_config()
#         config.update({"filters": self.filters})
#         return config


class Net:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Encode 共同部分
        self.conv1 = DoubleConvBN(64)
        self.pool1 = keras.layers.MaxPool2D()
        self.conv2 = DoubleConvBN(128)
        self.pool2 = keras.layers.MaxPool2D()
        self.conv3 = DoubleConvBN(256)
        self.pool3 = keras.layers.MaxPool2D()
        self.conv4 = DoubleConvBN(512)
        self.pool4 = keras.layers.MaxPool2D()
        self.conv5 = DoubleConvBN(1024)
        # Decode Unet部分
        self.tran_conv1 = keras.layers.Conv2DTranspose(512, kernel_size=2, strides=2)
        self.conv6 = DoubleConvBN(512)
        self.tran_conv2 = keras.layers.Conv2DTranspose(256, kernel_size=2, strides=2)
        self.conv7 = DoubleConvBN(256)
        self.tran_conv3 = keras.layers.Conv2DTranspose(128, kernel_size=2, strides=2)
        self.conv8 = DoubleConvBN(128)
        self.tran_conv4 = keras.layers.Conv2DTranspose(64, kernel_size=2, strides=2)
        self.conv9 = DoubleConvBN(64)
        self.conv10 = keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, activation='sigmoid')
        # Encode style transfer部分
        self.style_con1 = StyleConv(32)
        self.style_con2 = StyleConv(64)
        self.style_con3 = StyleConv(128)
        self.style_con4 = Residual(128, 3, 1)
        self.style_con5 = Residual(128, 3, 1)
        self.style_con6 = Residual(128, 3, 1)
        self.style_con7 = Residual(128, 3, 1)
        # Decode style transfer部分
        self.resize_conv1 = ResizeConv2d(64, 4)
        self.resize_conv2 = ResizeConv2d(32, 4)
        self.resize_conv3 = ResizeConv2d(32, 4)
        self.conv00 = keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='tanh')

    def build(self, training=None):
        x_input = keras.layers.Input(shape=self.input_shape)
        x = std00(x_input)
        l_input = tf.keras.layers.Input(shape=[1, ])
        labels = tf.keras.layers.Embedding(self.num_classes, 64)(l_input)
        labels2 = tf.keras.layers.Embedding(self.num_classes, 64)(l_input)
        # Encode 共同部分
        conv1 = self.conv1(x)  # 64 1
        pool = self.pool1(conv1)  # 64 1/2
        conv2 = self.conv2(pool)  # 128 1/2
        pool = self.pool2(conv2)  # 128 1/4
        conv3 = self.conv3(pool)  # 256 1/4
        pool = self.pool3(conv3)  # 256 1/8
        conv4 = self.conv4(pool)  # 512 1/8
        pool = self.pool4(conv4)  # 512 1/16
        y1 = self.conv5(pool)  # 1024 1/16
        # Decode部分
        y1 = self.tran_conv1(y1)  # 512 1/8
        y1 = keras.layers.concatenate([y1, conv4], axis=3)  # 1024 1/8
        y1 = self.conv6(y1)  # 512 1/8
        y1 = self.tran_conv2(y1)  # 256 1/4
        y1 = keras.layers.concatenate([y1, conv3], axis=3)  # 512 1/4
        y1 = self.conv7(y1)  # 256 1/4
        y1 = self.tran_conv3(y1)  # 128 1/2
        y1 = keras.layers.concatenate([y1, conv2], axis=3)  # 256 1/2
        y1 = self.conv8(y1)  # 128 1/2
        y1 = self.tran_conv4(y1)  # 64 1
        y1 = keras.layers.concatenate([y1, conv1], axis=3)  # 128 1
        y1 = self.conv9(y1)  # 64 1
        y1 = self.conv10(y1)  # 1 1

        # Encode style transfer部分
        d1 = self.style_con1(x)  # 32 1/2
        d2 = self.style_con2(d1)  # 64 1/4
        height = d2.get_shape()[1]
        width = d2.get_shape()[2]
        d2 = tf.multiply(labels, tf.reshape(d2, shape=(-1, height * width, 64)))
        d2 = tf.reshape(d2, shape=(-1, height, width, 64))
        d3 = self.style_con3(d2)  # 128 1/8
        d4 = self.style_con4(d3)
        d5 = self.style_con5(d4)
        d6 = self.style_con6(d5)
        d7 = self.style_con7(d6)
        u = self.resize_conv1(d7)  # 64 1/4
        u = keras.layers.concatenate([u, d2], axis=3)  # 128 1/4
        u = self.resize_conv2(u)  # 32 1/2
        u = keras.layers.concatenate([u, d1], axis=3)  # 64 1/2
        height = u.get_shape()[1]
        width = u.get_shape()[2]
        u = tf.multiply(labels2, tf.reshape(u, shape=(-1, height * width, 64)))
        u = tf.reshape(u, shape=(-1, height, width, 64))
        u = self.resize_conv3(u)  # 32 1
        u = self.conv00(u)

        y2 = (u + 1) * 127.5

        mask_G = 1. / (1 + tf.exp((0.5 - y1) * 32))
        y2 = tf.multiply(1 - mask_G, x_input) + tf.multiply(mask_G, y2)
        y = keras.layers.concatenate([y2, y1], axis=3)
        model0 = keras.models.Model([x_input, l_input], y)
        return model0


if __name__ == '__main__':
    net = Net((192, 256, 3), 7)
    model = net.build()
    model.summary()
