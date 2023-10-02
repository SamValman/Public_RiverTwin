#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  Marcin Luksza , MIT license
https://github.com/lukszamarcin/100-tiramisu-keras

Modified to TF2.X by Patrice Carbonneau
- uses tf.keras instead of keras
- re-added the maxpool from the Jegou et al original paper in the down tranistion, stride 1
- reshaped  to tile shape at output
- use elu instead of relu
- made a single conv_elu_bn with the bn at the start
- removed the l2 regularizer
- added a function to output a full tf model that can be called from another script
"""
import tensorflow as tf
from tensorflow.keras.layers import *
#from tensorflow.keras.regularizers import l2


# def relu_bn(x): return Activation('elu')(BatchNormalization(axis=-1)(x))


def conv(x, nf, sz, wd, p, stride=1):
    x = Conv2D(nf, (sz, sz), strides=(stride, stride), padding='same', kernel_initializer='he_uniform')(x)
    return Dropout(p)(x) if p else x


def conv_elu_bn(x, nf, sz=3, wd=0, p=0.2, stride=1):
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv2D(nf, (sz, sz), strides=(stride, stride),padding='same', kernel_initializer='he_uniform')(x)
    if p != 0.0:
        x = Dropout(p)(x)
    return x
    


def dense_block(n, x, growth_rate, p, wd):
    added = []
    for i in range(n):
        b = conv_elu_bn(x, growth_rate, p=p, wd=wd)
        x = Concatenate(axis=-1)([x, b])
        added.append(b)
    return x, added


def transition_dn(x, p, wd):
    # in the paper stride=1 but better results with stride=2
    x=conv_elu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=1)
    return MaxPooling2D((2,2))(x)#missing maxpool in github


def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = dense_block(n, x, growth_rate, p, wd)

        # keep track of skip connections
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added


def transition_up(added, wd=0):
    x = Concatenate(axis=-1)(added)
    _, r, c, ch = x.get_shape().as_list()
    return Conv2DTranspose(ch, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x)
#                 kernel_regularizer = l2(wd))(x)


def up_path(added, skips, nb_layers, growth_rate, p, wd):
    # use previously saved list of skip connections
    for i, n in enumerate(nb_layers):
        x = transition_up(added, wd)

        # concatenate the skip connections
        x = Concatenate(axis=-1)([x, skips[i]])
        x, added = dense_block(n, x, growth_rate, p, wd)
    return x


def reverse(a): return list(reversed(a))


def create_tiramisu(tile_size, nb_classes, img_input, nb_dense_block=6,
                    growth_rate=16, nb_filter=64, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4):

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips, added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)
    _, r, c, f = x.get_shape().as_list()
    x = Reshape((tile_size,tile_size, nb_classes))(x)
    return Activation('softmax')(x)


'''create and return the model'''
def tiramisu(tile_size, bands, Nclasses, summary=False):
    input_shape = (tile_size, tile_size, bands)
    img_input = Input(shape=input_shape)
    x = create_tiramisu(tile_size, Nclasses, img_input)
    model = tf.keras.Model(img_input, x)
    if summary:
        model.summary()
    return model
