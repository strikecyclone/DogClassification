# coding:UTF-8
'''
Created on 2017年 06月 28日 星期三 11:07:14 CST

@author: song
'''

import scipy.io as scio
from keras.utils.np_utils import to_categorical
import numpy as np
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from keras.utils.vis_utils import plot_model
from keras.callbacks import Callback
from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

# loss
class LossHistory(Callback):
    def __init__(self):
        Callback.__init__(self)
        self.losses = []
        self.accuracies = []

    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))

history = LossHistory()
# end

# let one image become 10 images, left-up/right-up/left-down/right-down/center and flip them
def createImages(ex_h,ex_w,path):
    list = np.empty([10, ex_h, ex_w, 3])
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img)
    img1 = cv2.merge([r,g,b])
    h,w,d = img1.shape
    img2 = cv2.flip(img1,1)

    sh = int(h*0.8)
    sw = int(w*0.8)
    hbh = int(h/10)
    hbw = int(w/10)

    img1_lu = img1[0:sh+1, 0:sw+1]
    img1_ru = img1[0:sh+1, w-sw:]
    img1_ld = img1[h-sh:, 0:sw+1]
    img1_rd = img1[h-sh:, w-sw:]
    img1_c = img1[hbh:hbh+sh+1, hbw:hbw+sw+1]
    img2_lu = img2[0:sh+1, 0:sw+1]
    img2_ru = img2[0:sh+1, w-sw:]
    img2_ld = img2[h-sh:, 0:sw+1]
    img2_rd = img2[h-sh:, w-sw:]
    img2_c = img2[hbh:hbh+sh+1, hbw:hbw+sw+1]

    list[0,:,:,:] = np.asarray(cv2.resize(img1_lu, (ex_w,ex_h)))
    list[1,:,:,:] = np.asarray(cv2.resize(img1_ru, (ex_w,ex_h)))
    list[2,:,:,:] = np.asarray(cv2.resize(img1_ld, (ex_w,ex_h)))
    list[3,:,:,:] = np.asarray(cv2.resize(img1_rd, (ex_w,ex_h)))
    list[4,:,:,:] = np.asarray(cv2.resize(img1_c, (ex_w,ex_h)))
    list[5,:,:,:] = np.asarray(cv2.resize(img2_lu, (ex_w,ex_h)))
    list[6,:,:,:] = np.asarray(cv2.resize(img2_ru, (ex_w,ex_h)))
    list[7,:,:,:] = np.asarray(cv2.resize(img2_ld, (ex_w,ex_h)))
    list[8,:,:,:] = np.asarray(cv2.resize(img2_rd, (ex_w,ex_h)))
    list[9,:,:,:] = np.asarray(cv2.resize(img2_c, (ex_w,ex_h)))
    return list
# end

# some constant
BASE_PATH = '/home/song/Documents/dog'
IMAGES_PATH = BASE_PATH + '/Images/'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
EPOCHS = 500
count = 0
# end

# load .mat file
train_mat = BASE_PATH + '/train_data.mat'
train_data = scio.loadmat(train_mat)

test_mat = BASE_PATH + '/test_data.mat'
test_data = scio.loadmat(test_mat)
# end

# make train_labels grow 10 times and convert to one-hot mode
print("labels start to grow 10 times......")

train_y = train_data['train_info'][0][0]['labels'] - 1.0
train_y_10 = np.empty([train_y.size*10])
for i in range(train_y.size):
    for j in range(0,10):
        train_y_10[i*10+j] = train_y[i]
train_labels_one_hot = to_categorical(train_y_10)

test_y = test_data['test_info'][0][0]['labels'] - 1.0
test_labels_one_hot = to_categorical(test_y)

print("OK!")
# end

vgg_model = VGG16(weights='imagenet', include_top=False)

# divide images into training and testing
print("start to read images......")

train_list = train_data['train_info'][0][0]['file_list']
train_x = np.empty([train_list.size*10, 7, 7, 512])
for i in range(train_list.size):
    cur_file_name = IMAGES_PATH + train_list[i][0][0]
    list = createImages(IMAGE_HEIGHT,IMAGE_WIDTH,cur_file_name)
    for j in range(0,10):
        cur_ex = np.expand_dims(list[j],axis=0)
        cur_input = preprocess_input(cur_ex)
        train_x[i*10+j,:] = vgg_model.predict(cur_input)
    if i%500 == 0:
        print("have read %d train_images"%i)

test_list = test_data['test_info'][0][0]['file_list']
test_x = np.empty([test_list.size, 7, 7, 512])
for j in range(test_list.size):
    cur_file_name = IMAGES_PATH + test_list[j][0][0]
    cur_file = image.load_img(cur_file_name, target_size=(IMAGE_WIDTH,IMAGE_HEIGHT))
    cur_file_mat = image.img_to_array(cur_file)
    cur_file_ex = np.expand_dims(cur_file_mat, axis=0)
    cur_input = preprocess_input(cur_file_ex)
    test_x[j,:] = vgg_model.predict(cur_input)
    if j%500 == 0:
        print("have read %d test_images"%j)

print("OK!")
# end

# model
model = Sequential()
model.add(Flatten(input_shape=(7,7,512)))
model.add(Dense(activation="relu", units=1024, kernel_initializer="uniform"))
model.add(Dropout(0.5))
model.add(Dense(activation="relu", units=1024, kernel_initializer="uniform"))
model.add(Dropout(0.5))
model.add(Dense(activation="softmax", units=120, kernel_initializer="uniform"))
# end

# write model to json
with open("dogs_model.json", "w") as f:
    f.write(model.to_json())

# plot model
plot_model(model, to_file="dogs_model.png", show_shapes=True)

# compile and fit
model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
model.fit(x=train_x, y=train_labels_one_hot , batch_size=32, epochs=EPOCHS, callbacks=[history])

# save model
model.save_weights("dogs_model_weights.hdf5")

# predict
predict_labels = model.predict_classes(test_x)

print(classification_report(test_y, predict_labels))
print(accuracy_score(test_y, predict_labels))
print(confusion_matrix(test_y, predict_labels))
