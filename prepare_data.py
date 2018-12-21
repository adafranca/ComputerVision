import matplotlib.pyplot as plt
import numpy as np
import cv2
from os import walk
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from imgaug import augmenters as iaa
import json


# # # Get path where dataset is stored # # #
dir_dataset = ['dataset-strabismus']
img_rows = 256
img_cols = 256


def random_rotation(image, mask, mask1):
    random_degree = random.uniform(-50, 50)
    return sk.transform.rotate(image, random_degree, preserve_range=True).astype(np.uint8), sk.transform.rotate(mask, random_degree, preserve_range=True).astype(np.uint8), sk.transform.rotate(mask1, random_degree, preserve_range=True).astype(np.uint8)


def random_noise(image_array: ndarray):
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(1.0, 3.0))
    ])
    return seq.augment_image(image_array)


def color(image_array: ndarray):
    seq = iaa.Sequential([
        iaa.Sharpen(lightness=1.00, alpha=1)
    ])
    return seq.augment_image(image_array)


def arithmetic(image_array: ndarray):
    seq = iaa.Sequential([
        iaa.Salt(p=0.03)
    ])
    return seq.augment_image(image_array)


def brightness(image):
    i = random.randint(0,3)
    items = [0.25, 1.00, 0.5, 1.5]
    seq = iaa.Sequential([
        iaa.Multiply(mul=items[i]),
        iaa.Sharpen(lightness=1.00, alpha=1)
    ])
    return seq.augment_image(image)


def horizontal_flip(image_array: ndarray, mask_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1], mask_array[:, ::-1]


def plot_im_mask(im,im_mask):
    ### Function to plot image mask

    im = np.array(im,dtype=np.uint8)
    im_mask = np.array(im_mask,dtype=np.uint8)
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(im_mask[:, :, 0])
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.bitwise_and(im,im,mask=im_mask))
    plt.axis('off')
    plt.show()


def readJSON():

    list_labels = []
    for (dirpath, dirnames, filenames) in walk(dir_dataset[0]):
        for file in filenames:
            if 'LABEL' in file and 'json' in file:
                with open(dir_dataset[0] + "/" + file) as f:
                    data = json.load(f)
                    list_labels.append(data)

    return list_labels


def dados():
    labels = readJSON()

    batch_images = np.zeros((np.shape(labels)[0] * 3, img_rows, img_cols, 3))
    batch_masks = np.zeros((np.shape(labels)[0] * 3,2, img_rows, img_cols, 1))

    z = 0

    for label in labels:
        img = cv2.imread(dir_dataset[0] + "/" + label['imagePath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channels = img.shape
        shapes = label['shapes']
        eyes = []
        iris = []

        mask1 = np.zeros((height, width, 3), np.uint8)
        for shape in shapes:
            points = shape['points']
            nome = shape['label']
            if nome == 'eyes':
                eyes.append(points)
            if nome == 'iris':
                iris.append(points)

        for point in eyes:
            point = np.array(point)
            mask1 = cv2.fillPoly(mask1, np.int32([point]), (0, 255, 255))
        for point in iris:
            point = np.array(point)
            mask1 = cv2.fillPoly(mask1, np.int32([point]), 255)

        mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2RGB)
        mask1 = cv2.resize(mask1, (img_cols, img_rows))
        # cv2.imwrite("new/" + label['imagePath'], img)
        # cv2.imwrite("new/MASK|" + label['imagePath'], mask1)
        img = cv2.resize(img, (img_cols, img_rows))
        img_mask = np.zeros_like(mask1[:, :, 0])
        iris = []
        for i in range(np.shape(mask1)[0]):
            for j in range(np.shape(mask1)[1]):
                r = str(mask1[i][j][0])
                g = str(mask1[i][j][1])
                b = str(mask1[i][j][2])

                # iris 0 , 0, 255
                if r == "0" and g == '0' and b == '255':
                  iris.append([i, j])

                # eyes 255 , 255, 0
                # if r == "255" and g == '255'   and b == '0':
                # img_mask[i, j] = 1

                if r != "0" or g != "0" or b != "0":
                    img_mask[i, j] = 1

        img_mask2 = np.zeros_like(mask1[:, :, 0])
        for l in iris:
            img_mask[l[0], l[1]] = 0
            img_mask2[l[0],l[1]] = 1

        img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))
        img_mask2 = np.reshape(img_mask2, (np.shape(img_mask2)[0], np.shape(img_mask2)[1], 1))

        plot_im_mask(img,img_mask)

        plot_im_mask(img,img_mask2)
        break
        masks = []

        masks.append(img_mask)
        masks.append(img_mask2)

        masks = np.array(masks)

        batch_images[z] = img
        batch_masks[z] = masks

        # augmented
        i = random.randint(0, 2)
        z += 1

        image, image_mask, image_mask2 = random_rotation(img, img_mask, img_mask2)

        masks_r = []
        masks_r.append(image_mask)
        masks_r.append(image_mask2)
        masks_r = np.array(masks_r)
        batch_images[z] = image
        batch_masks[z] = masks_r
        # plot_im_mask(image, image_mask)

        z += 1
        if i == 0:
            batch_images[z] = random_noise(img)
            batch_masks[z] = masks
        if i == 1:
            batch_images[z] = color(img)
            batch_masks[z] = masks
        if i == 2:
            batch_images[z] = brightness(img)
            batch_masks[z] = masks
        z += 1

    return batch_images, batch_masks


def createMask():
    labels = readJSON()

    batch_images = np.zeros((np.shape(labels)[0]*3, img_rows, img_cols, 3))
    batch_masks = np.zeros((np.shape(labels)[0]*3, img_rows, img_cols, 1))

    z = 0

    for label in labels:
        img = cv2.imread(dir_dataset[0] + "/" + label['imagePath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       # print(label)
        height, width, channels = img.shape
        shapes = label['shapes']
        eyes = []
        iris = []

        mask1 = np.zeros((height, width, 3), np.uint8)
        for shape in shapes:
            points = shape['points']
            nome = shape['label']
            if nome == 'eyes':
                eyes.append(points)
            if nome == 'iris':
                iris.append(points)

        for point in eyes:
            point = np.array(point)
            mask1 = cv2.fillPoly(mask1, np.int32([point]), (0, 255, 255))
        for point in iris:
            point = np.array(point)
            mask1 = cv2.fillPoly(mask1, np.int32([point]), 255)

        mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2RGB)
        mask1 = cv2.resize(mask1, (img_cols, img_rows))
        #cv2.imwrite("new/" + label['imagePath'], img)
        #cv2.imwrite("new/MASK|" + label['imagePath'], mask1)

        img = cv2.resize(img, (img_cols, img_rows))
        img_mask = np.zeros_like(mask1[:, :, 0])
        iris = []
        for i in range(np.shape(mask1)[0]):
            for j in range(np.shape(mask1)[1]):
                r = str(mask1[i][j][0])
                g = str(mask1[i][j][1])
                b = str(mask1[i][j][2])

                # iris 0 , 0, 255
               # if r == "0" and g == '0' and b == '255':
                  #  iris.append([i, j])

                # eyes 255 , 255, 0
                #if r == "255" and g == '255'   and b == '0':
                   #img_mask[i, j] = 1

                if r != "0" or g != "0" or b != "0":
                    img_mask[i,j] = 1

        #for l in iris:
        #    img_mask[l[0], l[1]] = 2

        img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))

        #plot_im_mask(mask1, img_mask)
        #break
        batch_images[z] = img
        batch_masks[z] = img_mask

        # augmented
        i = random.randint(0, 2)
        z += 1

        image, image_mask = random_rotation(img, img_mask)
        batch_images[z] = image
        batch_masks[z] = image_mask
        #plot_im_mask(image, image_mask)

        z+= 1
        if i == 0:
            batch_images[z] = random_noise(img)
            batch_masks[z] = img_mask
        if i == 1:
            batch_images[z] = color(img)
            batch_masks[z] = img_mask
        if i == 2:
            batch_images[z] = brightness(img)
            batch_masks[z] = img_mask
        z += 1


    return batch_images, batch_masks


def get_test():
    batch_images = np.zeros((1, img_rows, img_cols, 3))

    img = cv2.imread("dataset-strabismus/20140107_164742 001.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_cols, img_rows))
    batch_images[0] = img
    return batch_images


def showImage(img):
    screen_res = 1920, 1080
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









