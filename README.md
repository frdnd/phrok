# Phrok
This is a extension of [Erik Linder-Noréns minimal Deep Dream example](https://github.com/eriklindernoren/PyTorch-Deep-Dream/) which adds support for a COCO trained Resnet50, selection of individual neurons and offsets.
All images in [out_ref](https://github.com/frdnd/phrok/tree/master/out_ref) were rendered in 2560x1440 and downscaled by factor `0.25`, rendering one image takes about 15s on a RTX2070super.

## Installation
`pip3 install -r requirements.txt`

## Render an example
display all options with `python3 dream.py -h`

render the [cactus example](https://github.com/frdnd/phrok/blob/master/out_ref/cactus.jpg)

`python3 dream.py --from_example example_configs/cactus.yaml`

or equivalent

`python3 dream.py  --img_initialization "black" --iterations 30 --lr 0.1 --modeltype resnet50_coco --neuron 478 --octave_scale 1.4 --octaves 20 --offset 0.0001 --resnet_bnck 2 --resnet_sub_bnck 0`

which gives:

<p align="center">
    <img src="out_ref/cactus.jpg" width="640"\>
</p>

if you want to apply this to a input real input image use `python3 dream.py --img_initialization input_img --input_image imgs/frog.jpg --lr 0.01`

<p align="center">
    <img src="out_ref/frog.jpg" width="640"\>
</p>

## Parameter ranges
VGG19 can be expressed as a pure sequential model, due to this the only parameter needed is `layers` and `neurons`.
For Resnet50 I'm sticking to the model structure of the pytorch model for convenience, this model has 4 bottleneck layers (`resnet_bnk` parameter) which have
again several bottlenecks embedded (`resnet_sub_bnck`).

The number os neurons/channels for Resnet50 for each layer are:

| resnet_bnk | resnet_sub_bnck | neurons |
| :--------: | :------------:| :-----: |
| 0 | [0..2] | 256 |
| 1 | [0..3] | 512 |
| 2 | [0..5] | 1024 |
| 3 | [0..2] | 2048 |


