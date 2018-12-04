import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import cv2

from os import walk


# # # Get path where dataset is stored # # #

dir_dataset = ['dataset-strabismus']
img_rows = 128
img_cols = 128


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


def trans_image(image,bb_boxes_f,trans_range):
    # Translation augmentation
    bb_boxes_f = bb_boxes_f.copy(deep=True)

    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2

    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    rows,cols,channels = image.shape
    bb_boxes_f['xmin'] = bb_boxes_f['xmin']+tr_x
    bb_boxes_f['xmax'] = bb_boxes_f['xmax']+tr_x
    bb_boxes_f['ymin'] = bb_boxes_f['ymin']+tr_y
    bb_boxes_f['ymax'] = bb_boxes_f['ymax']+tr_y

    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

    return image_tr,bb_boxes_f

def stretch_image(img,bb_boxes_f,scale_range):
    # Stretching augmentation

    bb_boxes_f = bb_boxes_f.copy(deep=True)

    tr_x1 = scale_range*np.random.uniform()
    tr_y1 = scale_range*np.random.uniform()
    p1 = (tr_x1,tr_y1)
    tr_x2 = scale_range*np.random.uniform()
    tr_y2 = scale_range*np.random.uniform()
    p2 = (img.shape[1]-tr_x2,tr_y1)

    p3 = (img.shape[1]-tr_x2,img.shape[0]-tr_y2)
    p4 = (tr_x1,img.shape[0]-tr_y2)

    pts1 = np.float32([[p1[0],p1[1]],
                   [p2[0],p2[1]],
                   [p3[0],p3[1]],
                   [p4[0],p4[1]]])
    pts2 = np.float32([[0,0],
                   [img.shape[1],0],
                   [img.shape[1],img.shape[0]],
                   [0,img.shape[0]] ]
                   )

    M = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    img = np.array(img,dtype=np.uint8)

    bb_boxes_f['xmin'] = (bb_boxes_f['xmin'] - p1[0])/(p2[0]-p1[0])*img.shape[1]
    bb_boxes_f['xmax'] = (bb_boxes_f['xmax'] - p1[0])/(p2[0]-p1[0])*img.shape[1]
    bb_boxes_f['ymin'] = (bb_boxes_f['ymin'] - p1[1])/(p3[1]-p1[1])*img.shape[0]
    bb_boxes_f['ymax'] = (bb_boxes_f['ymax'] - p1[1])/(p3[1]-p1[1])*img.shape[0]

    return img,bb_boxes_f

def augment_brightness_camera_images(image):

    ### Augment brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def getImage(df, ind, size=(128, 128), augmentation = False, trans_range = 20, scale_range=20):
    file_name = df['frame'][ind]
    img = cv2.imread(dir_dataset[0] + '/' + file_name)
    img_size = np.shape(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    bnd_boxes = df[df['frame'] == file_name].reset_index()
    img_size_post = np.shape(img)

    if augmentation == True:
        img, bnd_boxes = trans_image(img, bnd_boxes, trans_range)
        img, bnd_boxes = stretch_image(img, bnd_boxes, scale_range)
        img = augment_brightness_camera_images(img)

    bnd_boxes['xmin'] = np.round(bnd_boxes['xmin'] / img_size[1] * img_size_post[1])
    bnd_boxes['xmax'] = np.round(bnd_boxes['xmax'] / img_size[1] * img_size_post[1])
    bnd_boxes['ymin'] = np.round(bnd_boxes['ymin'] / img_size[0] * img_size_post[0])
    bnd_boxes['ymax'] = np.round(bnd_boxes['ymax'] / img_size[0] * img_size_post[0])
    bnd_boxes['Area'] = np.round(bnd_boxes['xmax'] - bnd_boxes['xmin']) * (bnd_boxes['ymax'] - bnd_boxes['ymin'])

    return file_name, img, bnd_boxes

def generate_train_batch(df, batch_size = 32):

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))

    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(df))
            name_str,img,bb_boxes = getImage(df,i_line,
                                                   size=(img_cols, img_rows),
                                                  augmentation=True,
                                                   trans_range=50,
                                                   scale_range=50
                                                  )
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks


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
    plt.show()


def plot_bbox(bb_boxes,ind_bb,color='r',linewidth=2):
    ### Plot bounding box

    bb_box_i = [bb_boxes.iloc[ind_bb]['xmin'],
                bb_boxes.iloc[ind_bb]['ymin'],
                bb_boxes.iloc[ind_bb]['xmax'],
                bb_boxes.iloc[ind_bb]['ymax']]
    bb_box_i = list(map(int, bb_box_i))
    plt.plot([bb_box_i[0],bb_box_i[2],bb_box_i[2],
                  bb_box_i[0],bb_box_i[0]],
             [bb_box_i[1],bb_box_i[1],bb_box_i[3],
                  bb_box_i[3],bb_box_i[1]],
             color,linewidth=linewidth)


def plot_im_bbox(im,bb_boxes):
    ### Plot image and bounding box
    plt.imshow(im)
    for i in range(len(bb_boxes)):
        plot_bbox(bb_boxes,i,'g')

        bb_box_i = [bb_boxes.iloc[i]['xmin'],bb_boxes.iloc[i]['ymin'],
                bb_boxes.iloc[i]['xmax'],bb_boxes.iloc[i]['ymax']]
        plt.plot(bb_box_i[0],bb_box_i[1],'rs')
        plt.plot(bb_box_i[2],bb_box_i[3],'bs')
    plt.axis('off');


def dataset():
    df = xml2dataframe()
    training_gen = generate_train_batch(df, 12)
    return training_gen


def dataset_names():
    """ index, xmax, ymax, nome, tipo """

    df = xml2dataframe()
    elements_count = df.shape

    dic_name = dict()
    for index,row in df.iterrows():
        dic_name[row['frame']] = index

    batch_images = np.zeros((dic_name.items().__len__(), img_rows, img_cols, 3))
    batch_masks = np.zeros((dic_name.items().__len__(), img_rows, img_cols, 1))
    names = []
    idx = 0
    for value in dic_name:
         pos = dic_name[value]
         file_name, img, bnd_boxes = getImage(df, pos)
         img_mask = get_mask_seg(img, bnd_boxes)
         batch_images[idx] = img
         batch_masks[idx] = img_mask
         names.append(file_name)
         idx+=1


    return batch_images, batch_masks, names
