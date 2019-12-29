#!/usr/bin/env python3


import mobilenet_v3_keras as mnv3_keras
from tensorflow import keras

x = keras.Input((256, 256, 3))
m = mnv3_keras.create_mobilenet_v3(x)
m.save("foo", save_format="tf")
