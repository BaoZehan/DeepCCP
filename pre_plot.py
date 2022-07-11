# -*- coding: utf-8 -*-
# @Time : 2022/5/22 13:28
# @Author : Bao_Zehan
# @File : pre_plot.py
# @Project : text_conditional_local_style_transfer
import numpy as np
from Pre_Net import Net
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import imageio

net = Net((192, 256, 3), 7)
model = net.build()
model.load_weights('model_weights/pre_weights.h5')
path1 = 'inputs/1.jpg'
path2 = 'outtest'


def read_img(path):
    img_array = np.array(Image.open(path))
    return img_array


def style_out_only(img_path0, n, out_path0):
    img_0 = read_img(img_path0)
    if len(np.shape(img_0)) < 4:
        img_0 = tf.expand_dims(img_0, axis=0)
    pic_pre = img_0
    out_path0 = out_path0 + '/' + 'pre_' + str(n) + '.png'
    if n < 3:
        n = 3
    n = np.ceil(np.asarray(n / 3, dtype=np.float32) - 1)
    if 6 >= n >= 0:
        t = (np.ones(shape=(1,)) * n).astype(np.int32)
        pic_pre = model.predict([img_0, t])[:, :, :, 0:3]
    if 13 >= n > 6:
        t = (np.ones(shape=(1,)) * 6).astype(np.int32)
        pic_pre = model.predict([img_0, t])[:, :, :, 0:3]
        t = (np.ones(shape=(1,)) * (n - 7)).astype(np.int32)
        pic_pre = model.predict([pic_pre, t])[:, :, :, 0:3]
    if n >= 13:
        t = (np.ones(shape=(1,)) * 6).astype(np.int32)
        pic_pre = model.predict([img_0, t])[:, :, :, 0:3]
        t = (np.ones(shape=(1,)) * 6).astype(np.int32)
        pic_pre = model.predict([pic_pre, t])[:, :, :, 0:3]
    pic_pre = np.asarray(pic_pre[0], dtype=np.uint8)
    plt.imsave(out_path0, pic_pre, dpi=300)


def mask_out_only(img_path0, out_path0, img_name):
    img_path0 = img_path0 + '/' + img_name + '.jpg'
    img_0 = read_img(img_path0)
    out_path0 = out_path0 + '/' + img_name + '.png'
    if len(np.shape(img_0)) < 4:
        img_0 = tf.expand_dims(img_0, axis=0)
    t = (np.ones(shape=(1,))).astype(np.int32)
    mask_pre = model.predict([img_0, t])[:, :, :, 3:4]
    mask_pre = np.asarray(mask_pre[0], dtype=np.float32).repeat(3, axis=-1)
    plt.imsave(out_path0, mask_pre, dpi=300)


for i in range(0, 43, 3):
    style_out_only(path1, i, path2)

# path2 = 'outtest2'
# for i in [5,8,11,14,20,26,33]:
#     path1 = 'real'
#     mask_out_only(path1, path2, str(i))

gif_images = []
for i in range(0, 43, 3):
    img_paths = path2+'/pre_' + str(i) + '.png'
    gif_images.append(imageio.imread(img_paths))

imageio.mimsave(path2+"/test.gif", gif_images, fps=2)
