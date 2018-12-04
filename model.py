from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras import backend as K
from matplotlib import pyplot as plt
import cv2
import numpy as np


# PARAMETERS #

im_width = 128
im_height = 128

# UNET MODEL #


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet_pre_trained(input_img, n_filters=16, dropout=0.5, batchnorm=True):

    vgg_model = VGG16(weights='imagenet', input_tensor=input_img, include_top=False)

    layers = dict([(layer.name, layer) for layer in vgg_model.layers])

    vgg_top = layers['block5_conv3'].output
    # Now getting bottom layers for multi-scale skip-layers
    block1_conv2 = layers['block1_conv2'].output
    block2_conv2 = layers['block2_conv2'].output
    block3_conv3 = layers['block3_conv3'].output
    block4_conv3 = layers['block4_conv3'].output

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(vgg_top)
    u6 = concatenate([u6, block4_conv3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, block3_conv3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, block2_conv2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, block1_conv2], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.summary()
    return model


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.summary()
    return model


def train_generator(training_gen, samples_per_epoch=100, nb_epoch=5):

    input_img = Input((im_height, im_width, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    #model = get_unet_pre_trained(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(lr=1e-4), loss=IOU_calc_loss, metrics=[IOU_calc])
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    history = model.fit_generator(training_gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, callbacks=callbacks)


def train(x_train, y_train, x_valid, y_valid):
    input_img = Input((im_height, im_width, 3), name='img')
    model = get_unet_pre_trained(input_img, n_filters=16, dropout=0.05, batchnorm=True)

    model.compile(optimizer=Adam(lr=1e-4), loss=IOU_calc_loss, metrics=[IOU_calc])
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    results = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                        validation_data=(x_valid, y_valid))

    return model, results


def test(batch_img, model):
    pred_all = model.predict(batch_img)
    for i in range(20):
        im = np.array(batch_img[i], dtype=np.uint8)
        im_pred = np.array(255 * pred_all[i], dtype=np.uint8)

        rgb_mask_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
        rgb_mask_pred[:, :, 1:3] = 0 * rgb_mask_pred[:, :, 1:2]

        img_pred = cv2.addWeighted(rgb_mask_pred, 0.5, im, 0.5, 0)

        plt.figure(figsize=(8, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(im)
        plt.title('Original image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_pred)
        plt.title('Predicted segmentation mask')
        plt.axis('off')
        plt.show()


def iou_calc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_calc_loss(y_true, y_pred):
    return -iou_calc(y_true, y_pred)


def load_model():
    input_img = Input((im_height, im_width, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(optimizer=Adam(lr=1e-4), loss=iou_calc_loss(), metrics=[iou_calc]),
                  loss="binary_crossentropy", metrics=["accuracy"])

    model.load_weights('model-tgs-salt.h5')
    #model.evaluate(x_valid, y_valid, verbose=1)
    return model