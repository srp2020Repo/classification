from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2  #base_model



def build_MobileNetV2(IMG_SIZE):
    conv_base = MobileNetV2(weights=None,
                    include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    conv_base.trainable = True
    return model