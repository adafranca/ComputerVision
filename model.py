from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras import backend as K
from prepare_data import plot_im_mask
import cv2
import matplotlib.pyplot as plt
import numpy as np

# PARAMETERS #

im_width = 256
im_height = 256


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


def get_unet_pre_trained(input_img, n_filters=32, n_classes=2, dropout=0.5, batchnorm=True, activation='sigmoid', weights=None):

    vgg_model = VGG16(input_tensor=input_img, weights='imagenet',include_top=False)

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
    outputs = Conv2D(n_classes, (1, 1), activation=activation)(c9)
    model = Model(inputs=vgg_model.input, outputs=[outputs])

    for layer in model.layers[:18]:
        layer.trainable = False

    model.summary()
    return model


def train(x_train, y_train, x_valid, y_valid):
    input_img = Input((im_height, im_width, 3), name='img')
    model = get_unet_pre_trained(input_img, n_filters=32, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(lr=1e-4), loss=iou_calc_loss, metrics=[iou_calc])

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    results = model.fit(x_train, y_train, batch_size=5, epochs=150, callbacks=callbacks,
                        validation_data=(x_valid, y_valid))

    print(results.history['iou_calc'])


    return model, results


def test(batch_img, model):
    mask1 = model.predict(batch_img)
    masks = mask1[0]

    masks = np.reshape(masks, (2,256,256))
    for i in range(1):

        im = np.array(batch_img[i], dtype=np.uint8)
        img_mask = np.reshape( masks[0], (np.shape( masks[0])[0], np.shape( masks[0])[1], 1))
        img_mask1 = np.reshape(masks[1], (np.shape(masks[0])[0], np.shape(masks[0])[1], 1))
        im_pred1 = np.array(255 *img_mask1, dtype=np.uint8)
        im_pred = np.array(255 *img_mask, dtype=np.uint8)
        rgb_mask_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
        rgb_mask_pred[:, :, 1:3] = 0 * rgb_mask_pred[:, :, 1:2]
        img_pred = cv2.addWeighted(rgb_mask_pred, 0.5, im, 0.5, 0)

        iris = cv2.cvtColor(im_pred1, cv2.COLOR_GRAY2RGB)
        iris[:, :, 0] = 0 * iris[:, :, 0]
        iris[:, :, 2] = 0 * iris[:, :, 2]
        img_pred = cv2.addWeighted(iris, 0.5, img_pred, 0.5, 0)

        plot_im_mask(im, img_mask)
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
    model = get_unet_pre_trained(input_img, n_filters=32, dropout=0.05, batchnorm=True, n_classes=2)
    model.load_weights("model-tgs-salt.h5")
    return model


