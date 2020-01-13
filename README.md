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
achieves 67.36% (which I suppose you could interpret as paper's 67.4%) and
74.78%  top1 correspondingly. I was not able to achieve "paper" accuracies,
even though I'm pretty sure the code is correct. 

Training setup was as follows:

```python3
"large": {
		"1.0": {
				"alpha": 1.0,
				"dropout": 0.2,
				"batch_size": 512,
				"loss": "smooth_ce",
				"loss_kwargs": {"smoothing": 0.1},
				"epochs": 200,
				"optimizer": "sgd",
				"optimizer_kwargs": {
						"momentum": 0.9,
						"weight_decay": 2e-5,
						"nesterov": True
				},
				"scheduler": "cosine",
				"scheduler_kwargs": {
						"num_cycles": 1,
						"peak_lr": 0.8,
						"min_lr": 1e-7,
						"initial_warmup_step_fraction": 0.0,
						"cycle_warmup_step_fraction": 0.1,
				},
				"trainer_kwargs": {"use_ema": True},
		}
},
"small": {
		"1.0": {
				"alpha": 1.0,
				"dropout": 0.2,
				"batch_size": 1024,
				"epochs": 200,
				"loss": "smooth_ce",
				"loss_kwargs": {"smoothing": 0.1},
				"optimizer": "sgd",
				"optimizer_kwargs": {
						"momentum": 0.9,
						"weight_decay": 2e-5,
						"nesterov": True
				},
				"scheduler": "cosine",
				"scheduler_kwargs": {
						"num_cycles": 1,
						"peak_lr": 0.8,
						"min_lr": 1e-7,
						"initial_warmup_step_fraction": 0.0,
						"cycle_warmup_step_fraction": 0.1,
				},
				"trainer_kwargs": {"use_ema": True},
		}
},
```
Normal augmentation was changed to crop less aggressively with scale setting
within [0.2, 1.0] for faster convergence.

## Requirements

PyTorch implementation requires PyTorch 1.3.0+, Keras requires TF 2.0.0+.

## Contributions

Contributions are welcome. Please use Black to format the code.
