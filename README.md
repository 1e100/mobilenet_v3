# MobileNet V3

This repo contains two implementations of MobileNet V3: one for PyTorch and 
another for Keras (TF 2.0).

There's also a PyTroch checkpoint for "small" MobileNet V3 that achieves 66.718% 
top1 and 86.975% top5 accuracies on ImageNet. Training was done with SGD, lr 0.05, 
wd 1e-5, 150 epochs for initial training and then 2 50 epoch runs with lower peak 
LR. Cosine schedule with warmup.

Because TF is much slower than PyTorch, training for the Keras model is still
underway, but the model is straight-across port, so it should, hypothetically,
achieve the same accuracy results.
