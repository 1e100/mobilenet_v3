# MobileNet V3

This repo contains two implementations of MobileNet V3: one for PyTorch and
another for Keras (TF 2.0), including segmentation-specific variants. I have
used these for practical applications, they seem to work fine.

MobileNet V3 blocks in this implementation also retain the feature map if there
is a downsampling in the block so that the feature map can then be fed into a
detection or segmentation head. Keras version doesn't need this because there
you can get the relevant op outputs directly, but for PyTorch the way the paper
describes head attachment requires that you grab the output from inside the
block, so this helps you do it easily.

There's also a PyTroch checkpoint for "small" and "large" MobileNet V3 that
achieves 66.51% and 73.82%  top1 correspondingly. I was not able to achieve
"paper" accuracies, even though I'm pretty sure the code is correct. 

Training was done with SGD, lr 0.5, wd 1e-5, 300 epochs. Cosine schedule with
warmup.

## Contributions

Contributions are welcome. Please use Black to format the code.
