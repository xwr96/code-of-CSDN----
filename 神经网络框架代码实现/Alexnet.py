import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from skimage import transform
import warnings
from tqdm import tqdm
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten, Dropout
from keras.models import Model
from keras.utils import to_categorical
 
warnings.filterwarnings('ignore')
 
image_path = 'F:\\Keras_cnn\\images\\'  #文件路径这样写才正确，刚开始以为是'/'一直显示找不到该文件
IMG_HEIGHT = 400
IMG_WIDTH = 500
IMG_CHANNELS = 1
'''
    处理label
'''
train_csv = pd.read_csv('train.csv')
train_label_string = train_csv['species'].values
train_id = train_csv['id'].values
 
laber_number_dict = {}
train_label_number = []
number = 0
for i in train_label_string:
    if i in laber_number_dict:
        train_label_number.append(laber_number_dict[i])
    else:
        laber_number_dict.update({i: number})
        train_label_number.append(number)
        number += 1
 
id_label_dict = dict(zip(train_id, train_label_number))
 
test_csv = pd.read_csv('test.csv')
test_id = test_csv['id'].values
 
train_data = np.zeros((len(train_id), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
train_label = np.zeros((len(train_id), 1), dtype=np.uint8)
test_data = np.zeros((len(test_id), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
 
for n, i in tqdm(enumerate(train_id), total=len(train_data)):
    image_data = imread(image_path + str(i) + '.jpg')
    image_data = transform.resize(image_data, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    train_data[n] = image_data
    train_label[n] = id_label_dict[i]
 
train_label = to_categorical(train_label, 99)
 
for n, i in tqdm(enumerate(test_id), total=len(test_id)):
    image_data = imread(image_path + str(i) + '.jpg')
    image_data = transform.resize(image_data, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    test_data[n] = image_data
 
random_number = np.random.randint(len(train_id))
show_train_data = train_data[random_number].reshape(IMG_HEIGHT, IMG_WIDTH)
imshow(show_train_data)
plt.show()
 
'''
    开始搭建 AlexNet
'''
 
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
c1 = Conv2D(48, (11, 11), strides=4, activation='relu', kernel_initializer='uniform', padding='valid')(inputs)
c2 = BatchNormalization()(c1)
c3 = MaxPool2D((3, 3), strides=2, padding='valid')(c2)
 
c4 = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c3)
c5 = BatchNormalization()(c4)
c6 = MaxPool2D((3, 3), strides=2, padding='valid')(c5)
 
c7 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c6)
c8 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c7)
c9 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c8)
c10 = MaxPool2D((3, 3), strides=2, padding='valid')(c9)
 
c11 = Flatten()(c10)
c12 = Dense(256, activation='relu')(c11)  # 论文中是2048
c13 = Dropout(0.5)(c12)
 
c14 = Dense(256, activation='relu')(c13)  # 论文中是2048
c15 = Dropout(0.5)(c14)
outputs = Dense(99, activation='softmax')(c15)  # 论文中是1000
 
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
 
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-1.h5', verbose=1, save_best_only=True)
model.fit(train_data, train_label, validation_split=0.1, batch_size=256, epochs=256,
          callbacks=[earlystopper, checkpointer])
