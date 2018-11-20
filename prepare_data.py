import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from os import walk


# # # Get path where dataset is stored # # #

dir_dataset = ['dataset-strabismus']


def xml2dataframe():

    list_labels = []
    df_columns = ['nome', 'tipo', 'frame', 'xmin', 'ymin', 'xmax', 'ymax']
    df_labels = pd.DataFrame(columns=df_columns)

    for (dirpath, dirnames, filenames) in walk(dir_dataset[0]):
        for file in filenames:
            if 'LABEL' in file:
                list_labels.append(dirpath + '/' + file)
                parsedXML = ET.parse(dirpath + "/" + file)

                for root in parsedXML.getroot():
                    if 'object' in root.tag:
                        name = root.find('name').text
                        frame = file.replace('LABEL|', '') + ".jpg"
                        bndbox = root.find('bndbox')
                        xmin = float(bndbox.find('xmin').text.replace(".0", ''))
                        ymin = float(bndbox.find('ymin').text.replace(".0", ''))
                        xmax = float(bndbox.find('xmax').text.replace(".0", ''))
                        ymax = float(bndbox.find('ymax').text.replace(".0", ''))
                        df_labels = df_labels.append(
                            pd.Series([file, name, frame, xmin, ymin, xmax, ymax], index=df_columns), ignore_index=True)

    return df_labels


def showImage(img):
    screen_res = 1280, 720
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


def get_mask_seg(img,bb_boxes_f):

    #### Get mask

    img_mask = np.zeros_like(img[:,:,0])
    for i in range(len(bb_boxes_f)):
        bb_box_i = [bb_boxes_f.iloc[i]['xmin'],bb_boxes_f.iloc[i]['ymin'],
                bb_boxes_f.iloc[i]['xmax'],bb_boxes_f.iloc[i]['ymax']]
        bb_box_i = list(map(int, bb_box_i))
        img_mask[bb_box_i[1]:bb_box_i[3],bb_box_i[0]:bb_box_i[2]] = 1
        img_mask = np.reshape(img_mask,(np.shape(img_mask)[0],np.shape(img_mask)[1],1))

    return img_mask


def plot_im_mask(im,im_mask):
    ### Function to plot image mask

    im = np.array(im,dtype=np.uint8)
    im_mask = np.array(im_mask,dtype=np.uint8)
    plt.subplot(1,3,1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(im_mask[:,:,0])
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(cv2.bitwise_and(im,im,mask=im_mask));
    plt.axis('off')
    plt.show();


def getImage(df, ind, size=(1290,1080), augmentation=False, trans_range=20, scale_range=20):
    file_name = df['frame'][ind]
    img = cv2.imread(dir_dataset[0] + '/' + file_name)
    img_size = np.shape(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    bnd_boxes = df[df['frame'] == file_name].reset_index()
    img_size_post = np.shape(img)

    bnd_boxes['xmin'] = np.round(bnd_boxes['xmin'] / img_size[1] * img_size_post[1])
    bnd_boxes['xmax'] = np.round(bnd_boxes['xmax'] / img_size[1] * img_size_post[1])
    bnd_boxes['ymin'] = np.round(bnd_boxes['ymin'] / img_size[0] * img_size_post[0])
    bnd_boxes['ymax'] = np.round(bnd_boxes['ymax'] / img_size[0] * img_size_post[0])
    bnd_boxes['Area'] = np.round(bnd_boxes['xmax'] - bnd_boxes['xmin']) * (bnd_boxes['ymax'] - bnd_boxes['ymin'])

    return file_name, img, bnd_boxes


def init():

    df = xml2dataframe()
    file_name, img, bnd_boxes = getImage(df, 6)
    img_mask = get_mask_seg(img, bnd_boxes)
    plot_im_mask(img, img_mask)
init()