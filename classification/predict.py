import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

import os
import matplotlib.pyplot as plt
import numpy as np

from model.ResNet.resnet_50_v2 import build_ResNet50V2



#config
batch_size = 8
IMG_SIZE = 300
learning_rate = 2e-5
checkpoint_save_path = "checkpoint/ResNet/ResNet50v2.ckpt"  #MANUALLY MODIFY CHECKPOINT_PATH

#File path
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
model = build_ResNet50V2(IMG_SIZE) # BUILD MODEL

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


#predict
pred = model.predict(validation_generator)
pred = np.argmax(pred, axis=1)
print(len(pred))
np.save('model/ResNet/ResNet50v2_pred.npy', pred)

'''
#predictions as per filename
labels = (validation_generator.class_indices)
label = dict((v,k) for (k,v) in labels.items())

predictions = [label[i] for i in pred]

filenames = validation_generator.filenames
for idx in range(len(filenames)):
    print('predict: ', predictions[idx])
    print('filepath: ', filenames[idx])
    print('')
'''

#accuracy

num = 0
for i in range(len(validation_generator.filenames)):
    if(pred[i] == validation_generator.labels[i]):
        num += 1
print(num / total_val)
