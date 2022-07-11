# -*- coding: utf-8 -*-
# @Time : 2022/7/6 11:00
# @Author : Bao_Zehan
# @File : Loss_Net.py
# @Project : DeepCCP
import typing
import tensorflow as tf

CONTENT_LAYERS = {'block4_conv2': 0.5, 'block5_conv2': 0.5}
STYLE_LAYERS = {'block1_conv1': 2e-01, 'block2_conv1': 5e-03, 'block3_conv1': 8e-04, 'block4_conv1': 6e-05,
                'block5_conv1': 1e-03}


def get_vgg19_model(layers):

    vgg = tf.keras.applications.VGG19(include_top=False, weights=None)
    vgg.load_weights('model_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    outputs = [vgg.get_layer(layer).output for layer in layers]
    model = tf.keras.Model([vgg.input, ], outputs)
    model.trainable = False
    return model


class NeuralStyleTransferModel(tf.keras.Model):

    def __init__(self, content_layers: typing.Dict[str, float] = CONTENT_LAYERS,
                 style_layers: typing.Dict[str, float] = STYLE_LAYERS):
        super(NeuralStyleTransferModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
        self.outputs_index_map = dict(zip(layers, range(len(layers))))
        self.vgg = get_vgg19_model(layers)

    def call(self, inputs, training=None, mask=None):
        outputs = self.vgg(inputs)
        content_outputs = []
        for layer, factor in self.content_layers.items():
            content_outputs.append((outputs[self.outputs_index_map[layer]], factor))
        style_outputs = []
        for layer, factor in self.style_layers.items():
            style_outputs.append((outputs[self.outputs_index_map[layer]], factor))
        return {'content': content_outputs, 'style': style_outputs}