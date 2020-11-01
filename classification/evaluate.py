import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix

import os
import matplotlib.pyplot as plt
import numpy as np

from model.Inception.inception_v3 import build_InceptionV3
from utils.plot_confusion_matrix import plot_confusion_matrix

#config
batch_size = 8
IMG_SIZE = 300
learning_rate = 2e-5
checkpoint_save_path = "checkpoint/Inception/Inceptionv3.ckpt"  #MANUALLY MODIFY CHECKPOINT_PATH

validation_dir = 'dataset/validation'
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
        batch_size=batch_size,
        class_mode='sparse')


total_val = len(validation_generator.filepaths)

#model
model = build_InceptionV3(IMG_SIZE) # BUILD MODEL

model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizers.Adam(lr=learning_rate),
              metrics=['sparse_categorical_accuracy'])



#checkpoint
if os.path.exists(checkpoint_save_path + '.index'):
    print('--------------------load the model------------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)



#evaluate
score = model.evaluate(validation_generator)
print('Loss(sparse_categorical_loss): ', score[0])
print('ACC(sparse_categorical_accuracy): ', score[1])

#猫和狗两种标签，存入到labels中
labels = ['gctb','other']

# 预测验证集数据整体准确率
Y_pred = model.predict(validation_generator)
# 将预测的结果转化为one hot向量
Y_pred_classes = np.argmax(Y_pred, axis = 1)
# 计算混淆矩阵
confusion_mtx = confusion_matrix(y_true = validation_generator.labels,y_pred = Y_pred_classes)
# 绘制混淆矩阵
plot_confusion_matrix(confusion_mtx, savepath='model/Inception', normalize=True, target_names=labels)
