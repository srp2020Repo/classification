import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

import os
import matplotlib.pyplot as plt
import numpy as np

from model.Incep_Res.inception_resnet_v2 import build_InceptionResNetV2
from model.MobileNet.mobilenet_v2 import build_MobileNetV2
from model.VGG.vgg19 import build_VGG19
from model.Xception.xception import build_Xception

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#config
batch_size = 64
epochs = 40
IMG_SIZE = 224
learning_rate = 2e-5
#checkpoint_list = ['checkpoint_norm/IncepRes/IncepResv2.ckpt', 'checkpoint/MobileNet/MobileNetv2.ckpt', 'checkpoint/VGG/VGG19.ckpt', 'checkpoint/XCeption/Xception.ckpt']
checkpoint_save_path = 'checkpoint/IncepRes/IncepResv2.ckpt'  #MANUALLY MODIFY CHECKPOINT_PATH
#figure_list = ['Figure_norm/acc_loss_incepResv2.png', 'Figure/acc_loss_mobilenetv2.png', 'Figure/acc_loss_vgg19.png', 'Figure/acc_loss_xception.png']
figure_save_path = 'Figure/acc_loss_incepResv2_3.png'


#File path
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

#Data Augmentation 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
test_datagen = ImageDataGenerator(rescale=1./255)

#Data generator
train_generator = train_datagen.flow_from_directory(
      train_dir,
      target_size=(IMG_SIZE, IMG_SIZE),
      shuffle=True,
      batch_size=batch_size,
      class_mode='sparse')

 
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
        batch_size=batch_size,
        class_mode='sparse')

total_train = len(train_generator.filepaths)
total_val = len(validation_generator.filepaths)

#model

model = build_InceptionResNetV2(IMG_SIZE) # BUILD MODEL

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



#train
history = model.fit_generator(
      train_generator,
      steps_per_epoch = total_train // batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps = total_val // batch_size,
      callbacks = [cp_callback],
      verbose=1)


#plot
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(figure_save_path)
#plt.show()

