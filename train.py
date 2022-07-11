# -*- coding: utf-8 -*-
# @Time : 2022/7/6 10:58
# @Author : Bao_Zehan
# @File : train.py
# @Project : DeepCCP
import numpy as np
from tensorflow import keras
from Pre_Net import Net
import tensorflow as tf
from Loss_Net import NeuralStyleTransferModel
import matplotlib.pyplot as plt
import os
import pickle

CONTENT_LOSS_FACTOR1 = 1
CONTENT_LOSS_FACTOR2 = 40

STYLE_LOSS_FACTOR = 1000
TV_FACTOR = 0.01

BG_FACTOR = 0.1

WIDTH = 256

HEIGHT = 192
BS = 32
Epo = 80

img_0 = np.load('data/train_all/img_train.npy')
target_0 = np.load('data/train_all/target_train.npy')
mask_img_0 = np.expand_dims(np.load('data/train_all/mask_img_train.npy'), axis=-1)
mask_target_0 = np.expand_dims(np.load('data/train_all/mask_target_train.npy'), axis=-1)
label_0 = np.load('data/train_all/label_train.npy')
dict0 = {}
for key in label_0:
    dict0[key] = dict0.get(key, 0) + 1

img_1 = np.load('data/val_all/img_val.npy')
target_1 = np.load('data/val_all/target_val.npy')
mask_img_1 = np.expand_dims(np.load('data/val_all/mask_img_val.npy'), axis=-1)
mask_target_1 = np.expand_dims(np.load('data/val_all/mask_target_val.npy'), axis=-1)
label_1 = np.load('data/val_all/label_val.npy')
dict1 = {}
for key in label_1:
    dict1[key] = dict1.get(key, 0) + 1

TNum = img_0.shape[0]
VNum = img_1.shape[0]
# T_step = int(TNum / BS)
# V_step = int(VNum / BS)
T_step = 50
V_step = 20

image_mean = tf.constant([131.4479, 123.1393, 80.2495], dtype=tf.float32)
image_std = tf.constant([62.7189, 53.4266, 63.5227], dtype=tf.float32)
vgg_mean = [103.939, 116.779, 123.68]
img_2 = np.load('data/test_all/img_test.npy')
target_2 = np.load('data/test_all/target_test.npy')
mask_img_2 = np.expand_dims(np.load('data/test_all/mask_img_test.npy'), axis=-1)
mask_target_2 = np.expand_dims(np.load('data/test_all/mask_target_test.npy'), axis=-1)
label_2 = np.load('data/test_all/label_test.npy')
img2 = img_2[12]


def std00(x):
    x = tf.cast(x, tf.float32)
    x = tf.subtract(x, image_mean)
    x = tf.divide(x, image_std)
    return x


def std01(x):
    red, green, blue = tf.split(x * 1.0, num_or_size_splits=3, axis=3)
    bgr = tf.concat(axis=3, values=[
        blue - vgg_mean[0],
        green - vgg_mean[1],
        red - vgg_mean[2],
    ])
    return bgr / 255.


def gray0(x):
    red, green, blue = tf.split(x * 1.0, num_or_size_splits=3, axis=3)
    gray = tf.maximum(red, green)
    bgr = tf.concat(axis=3, values=[
        gray - vgg_mean[0],
        gray - vgg_mean[1],
        gray - vgg_mean[2],
    ])
    return bgr / 255.


def show_img(img):
    # 展示一张图片
    plt.imshow(img)
    plt.show()


def generator0(img, target, mask_img, mask_target, label, batch_size):
    while True:
        index = []
        t = 0
        for i0 in dict0:
            index = np.append(index, np.random.choice(range(t, t + dict0[i0]), batch_size, replace=False))
            t = t + dict0[i0]
        index = np.random.choice(np.asarray(index, dtype=np.int32), batch_size, replace=False)
        yield [np.asarray(img[index], dtype=np.float32), label[index]], np.concatenate([
            np.asarray(img[index], dtype=np.float32),
            np.asarray(target[index], dtype=np.float32),
            np.asarray(mask_img[index], dtype=np.float32),
            np.asarray(mask_target[index], dtype=np.float32)], axis=-1)


def generator1(img, target, mask_img, mask_target, label, batch_size):
    while True:
        index = []
        t = 0
        for i0 in dict1:
            index = np.append(index, np.random.choice(range(t, t + dict1[i0]), batch_size, replace=False))
            t = t + dict1[i0]
        index = np.random.choice(np.asarray(index, dtype=np.int32), batch_size, replace=False)

        yield [np.asarray(img[index], dtype=np.float32), label[index]], np.concatenate([
            np.asarray(img[index], dtype=np.float32),
            np.asarray(target[index], dtype=np.float32),
            np.asarray(mask_img[index], dtype=np.float32),
            np.asarray(mask_target[index], dtype=np.float32)], axis=-1)


model_style = NeuralStyleTransferModel()


def _compute_content_loss(noise_features, target_features):
    l2 = tf.square(noise_features - target_features)
    content_loss = tf.reduce_sum(l2)
    x = tf.size(noise_features)
    content_loss = content_loss / (tf.cast(x, tf.float32) * 4.)
    return content_loss


def compute_content_loss(noise_content_feature, target_content_features):
    content_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_feature, target_content_features):
        for i0 in range(BS):
            layer_content_loss = _compute_content_loss(noise_feature[i0, :, :, :], target_feature[i0, :, :, :])
            content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)


def gram_matrix(feature, mask_0):
    feature = tf.multiply(feature, mask_0)
    x = tf.transpose(feature, perm=[2, 0, 1])
    x = tf.reshape(x, (x.shape[0], -1))
    # 计算x和x的逆的乘积
    return x @ tf.transpose(x)


def _compute_style_loss(noise_feature, mask_0, target_feature, mask_1):
    noise_gram_matrix = gram_matrix(noise_feature, mask_0) / (tf.sqrt(tf.reduce_mean(mask_0)))
    style_gram_matrix = gram_matrix(target_feature, mask_1) / (tf.sqrt(tf.reduce_mean(mask_1)))
    l2 = tf.square(noise_gram_matrix - style_gram_matrix)
    style_loss = tf.reduce_sum(l2)
    size = tf.size(target_feature)
    x = tf.cast(size, tf.float32)
    return style_loss / (x ** 2)


def compute_mask_style_loss(img_G_style_feature, target_style_feature, mask_0, mask_1):
    height = mask_0.get_shape()[1]
    width = mask_0.get_shape()[2]
    new_height = height
    new_width = width
    style_loss2 = []
    for (img_G_feature, factor), (target_feature, _) in zip(img_G_style_feature, target_style_feature, ):
        mask_00 = tf.image.resize(mask_0, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_10 = tf.image.resize(mask_1, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        for i0 in range(BS):
            layer_style_loss = _compute_style_loss(img_G_feature[i0, :, :, :], mask_00[i0, :, :, :],
                                                   target_feature[i0, :, :, :], mask_10[i0, :, :, :])
            style_loss2.append(layer_style_loss * factor)
        new_height = new_height // 2
        new_width = new_width // 2
    return tf.reduce_sum(style_loss2)


def style_content_loss(img_G, content_image0, style_image0, mask_img_G, mask_style):
    img_G_features = model_style(std01(img_G))
    img_G_gray_features = model_style(gray0(img_G))['content']
    content_features_gray = model_style(gray0(content_image0))['content']
    content_features = model_style(std01(content_image0))['content']
    style_features = model_style(std01(style_image0))['style']
    content_loss1 = compute_content_loss(img_G_gray_features, content_features_gray)
    content_loss2 = compute_content_loss(img_G_features['content'], content_features)
    style_loss = compute_mask_style_loss(img_G_features['style'], style_features, mask_img_G, mask_style)

    return CONTENT_LOSS_FACTOR1 * tf.sqrt(tf.square(content_loss1 - 0.05)) + CONTENT_LOSS_FACTOR2 * tf.sqrt(
        tf.square(content_loss2 - 0.05)) + style_loss * STYLE_LOSS_FACTOR


def dice_loss(content_mask, mask_img_G):
    y_true_f = keras.backend.flatten(content_mask)
    y_pre_f = keras.backend.flatten(mask_img_G)
    intersection = keras.backend.sum(y_true_f * y_pre_f)
    area_true = keras.backend.sum(y_true_f * y_true_f)
    area_pre = keras.backend.sum(y_pre_f * y_pre_f)
    dice = (2. * intersection + 1.) / (area_true + area_pre + 1.)
    return 1. - dice


def tv_loss(img_G, mask_img_G):
    tv = tf.reduce_mean(tf.multiply(tf.square(img_G[:, :, 1:, :] - img_G[:, :, :-1, :]), mask_img_G[:, :, 1:, :])) + \
         tf.reduce_mean(tf.multiply(tf.square(img_G[:, 1:, :, :] - img_G[:, :-1, :, :]), mask_img_G[:, 1:, :, :]))
    return tv / BS


def bg_loss(img_G, mask_img_G, content_img):
    mask_img_G = 1 - mask_img_G
    loss = tf.reduce_mean(tf.square(tf.multiply(mask_img_G, (img_G - content_img))))
    return loss


def total_loss(y_true, y_pre):
    loss = style_content_loss(y_pre[:, :, :, 0:3], y_true[:, :, :, 0:3], y_true[:, :, :, 3:6],
                              y_pre[:, :, :, 3:4], y_true[:, :, :, 7:8]) / BS
    loss = loss + tv_loss(y_pre[:, :, :, 0:3], y_pre[:, :, :, 3:4]) * TV_FACTOR
    loss = loss + bg_loss(y_pre[:, :, :, 0:3], y_pre[:, :, :, 3:4], y_true[:, :, :, 0:3]) * BG_FACTOR / BS
    return loss


def total_loss_Unet(y_true, y_pre):
    loss = dice_loss(y_pre[:, :, :, 3:4], y_true[:, :, :, 6:7])
    return loss


def IoU(y_true, y_pre):
    y_true_f = keras.backend.flatten(y_true[:, :, :, 6:7])
    y_pre = tf.where(y_pre[:, :, :, 3:4] < 0.5, 0., 1.)
    y_true_f = tf.cast(y_true_f, dtype=tf.float32)
    y_pre_f = keras.backend.flatten(y_pre)
    intersection = keras.backend.sum(y_true_f * y_pre_f)
    union = keras.backend.sum(tf.where((y_true_f + y_pre_f) > 0, 1., 0.))
    IoU0 = intersection / (union + 1e-10)
    return IoU0


def tf_psnr(y_true, y_pre):
    img_G = y_pre[:, :, :, 0:3] / 255.
    img_content = y_true[:, :, :, 0:3] / 255.
    return tf.image.psnr(img_G, img_content, max_val=1)


def _tv_loss(y_true, y_pre):
    return tv_loss(y_pre[:, :, :, 0:3], y_pre[:, :, :, 3:4])


def content_coef1(y_true, y_pre):
    img_G_features = model_style(gray0(y_pre[:, :, :, 0:3]))['content']
    content_features = model_style(gray0(y_true[:, :, :, 0:3]))['content']
    return compute_content_loss(img_G_features, content_features) / BS


def content_coef2(y_true, y_pre):
    img_G_features = model_style(std01(y_pre[:, :, :, 0:3]))['content']
    content_features = model_style(std01(y_true[:, :, :, 0:3]))['content']
    return compute_content_loss(img_G_features, content_features) / BS


def _style_content_loss(img_G, content_image0, style_image0, mask_img_G, mask_style):
    img_G_features = model_style(std01(img_G))
    style_features = model_style(std01(style_image0))['style']
    style_loss = compute_mask_style_loss(img_G_features['style'], style_features, mask_img_G, mask_style)
    return style_loss / BS


def style_coef(y_true, y_pre):
    loss = _style_content_loss(y_pre[:, :, :, 0:3], y_true[:, :, :, 0:3], y_true[:, :, :, 3:6],
                               y_pre[:, :, :, 3:4], y_true[:, :, :, 7:8])
    return loss


# 模型的创建
net = Net((192, 256, 3), 7)
model = net.build()


def pre(x0, n):
    if len(np.shape(x0)) < 4:
        x0 = tf.expand_dims(x0, axis=0)
    t = (np.ones(shape=(1,)) * n).astype(np.int32)
    x0 = model.predict([x0, t[0:1]])[:, :, :, 0:3]
    x0 = np.asarray(x0, dtype=np.uint8)
    return x0


def pre_mask(x0):
    if len(np.shape(x0)) < 4:
        x0 = tf.expand_dims(x0, axis=0)
    t = (np.ones(shape=(1,)) * 1).astype(np.int32)
    x0 = model.predict([x0, t[0:1]])[:, :, :, 3:4]
    return x0


# tf.config.experimental_run_functions_eagerly(True)
# Unet
# for layers in model.layers:
#     layers.trainable = False
# for i in range(4, 12):
#     model.layers[i].trainable = True
# for i in [13,16,21,23,27,29,33,35,39,41]:
#     model.layers[i].trainable = True
# model.compile(optimizer='Adam', loss=total_loss_Unet, metrics=[IoU])

# style
model.load_weights('model_weights/seg_weights.h5')
for layers in model.layers:
    layers.trainable = True
for i in range(4, 12):
    model.layers[i].trainable = False
for i in [13, 16, 21, 23, 27, 29, 33, 35, 39, 41]:
    model.layers[i].trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(lr=2.5e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    loss=total_loss, metrics=[tf_psnr, content_coef1, content_coef2, style_coef, ])

save_dir = 'pre_weights'
filepath = "pre_weights_{epoch:03d}.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, filepath),
                                                monitor='val_loss',
                                                save_weights_only=True
                                                )


class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs: None):
        for i in [0, 6]:
            t1 = pre(img2, i)
            show_img(t1[0])
        t1 = pre(t1[0], 2)
        show_img(t1[0])
        pass


history = model.fit_generator(generator0(img_0, target_0, mask_img_0, mask_target_0, label_0, BS),
                              steps_per_epoch=T_step,
                              epochs=Epo,
                              validation_data=generator1(img_1, target_1, mask_img_1, mask_target_1, label_1, BS),
                              validation_steps=V_step,
                              callbacks=[MyCallback(), checkpoint]
                              )

with open('model_weights/logs/log.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)

for layers in model.layers:
    layers.trainable = True
model.save_weights('pre_weights/pre_weights_weights.h5')
