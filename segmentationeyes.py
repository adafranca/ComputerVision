from prepare_data import generate_train_batch, getdatasetready
from keras import backend as K
from sklearn.model_selection import train_test_split
import model

num_classes = 1

# save and compute_metrics
#vis = VIS(save_path=opt.checkpoint_path)


im_height = 1096
im_width = 980
smooth = 1.
training_gen = generate_train_batch(1)

#train_generator, train_samples = dataLoader(getdatasetready(), opt.batch_size,(im_height, im_width), mean=dataset_mean, std=dataset_std)
#test_generator, test_samples = dataLoader(opt.data_path+'/val/', 1,  (im_height, im_width), train_mode=False,mean=dataset_mean, std=dataset_std)

def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)


def main():
        #input_img = Input((im_height, im_width, 1), name='img')
        #train_model(training_gen,training_gen,257, checkpoint_path="weights")
        x, y = getdatasetready()
        X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.15, random_state=2018)
        model.train(X_train, y_train, X_valid, y_valid)

main()