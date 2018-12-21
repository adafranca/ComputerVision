from prepare_data import get_test, createMask, plot_im_mask, dados
from model import get_unet_pre_trained, train, test, load_model
import numpy as np
import cv2


def video_ts(model):
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()
        cv2.imshow("name", frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        batch_images = np.zeros((1, 256, 256, 3))
        batch_images[0] = img
        test(batch_images, model)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def main():
        '''x, y = dados()
        #input_img = Input((384, 512, 3), name='img')
        # x, y = get_images()
        X = x[0: 260]
        Y = y[0: 260]
        x = x[:-260]
        y = y[:-260]
        X_valid = x[0:132]
        Y_valid = y[0:132]
        x_t = x[0:-132]
        y_t = y[0:-132]'''

        # print(np.shape(X_valid))
        #plot_im_mask(x_t[0], y_t[0])
        testa = get_test()
        #model = get_unet_pre_trained(input_img, n_filters=32, dropout=0.05, batchnorm=True)
        #model_trained, results = train(X, Y, X_valid, Y_valid)
        model_trained = load_model()
        #video_ts(model_trained)
        test(testa, model_trained)

main()


