#!/usr/bin/env python3

import unittest
import tempfile

import mobilenet_v3_keras as mnv3_keras
from tensorflow import keras


class TestKerasMobileNetV3(unittest.TestCase):

    # Minimal smoke test: verify the model can be created and saved.
    def test_create_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            x = keras.Input((256, 256, 3))
            m = mnv3_keras.create_mobilenet_v3(x)
            m.save(d, save_format="tf")


if __name__ == "__main__":
    unittest.main()
