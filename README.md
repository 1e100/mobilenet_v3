# MobileNet V3

This repo contains two implementations of MobileNet V3: one for PyTorch and 
another for Keras (TF 2.0).

There's also a PyTroch checkpoint for "small" and "large" MobileNet V3 that
achieves 66.51% and 73.82%  top1 correspondingly. I was not able to achieve
"paper" accuracies, even though I'm pretty sure the code is correct. 

Training was done with SGD, lr 0.5, wd 1e-5, 300 epochs. Cosine schedule with
warmup.

## Contributions

Contributions are welcome. Please use Black to format the code.
