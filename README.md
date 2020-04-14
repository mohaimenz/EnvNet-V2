# EnvNet-V2
Tensorflow implementation of between-class learning for sound recognition https://arxiv.org/abs/1711.10282

This repository is just a translation of the chainer implementation of EnvNet-V2 to Keras/Tensorflow. The chainer implementation can be found here at: https://github.com/mil-tokyo/bc_learning_sound

## Setup
- Install Tensorflow 2.0
- Prepare datasets following [this page](https://github.com/mil-tokyo/bc_learning_sound/tree/master/dataset_gen).


## Training
- Template:
		python main.py --dataset [esc50, esc10, or urbansound8k] --netType [envnet or envnetv2] --data path/to/dataset/directory/ (--BC) (--strongAugment)
