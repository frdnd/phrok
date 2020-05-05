import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
from PIL import Image as pil
import argparse
import os
import tqdm
import scipy.ndimage as nd
from utils import deprocess, preprocess, clip
import yaml
from os.path import basename, splitext


def dream(image, model, iterations, lr, neuron, offset, *args, **kwargs):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)

        if neuron is None:
            loss = out.norm()
        else:
            loss = out[:, neuron, :, :].norm()

        loss.backward()
        avg_grad = max(np.abs(image.grad.data.cpu().numpy()).mean(), offset)
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, octaves, neuron, offset, *args, **kwargs):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave

    img_octaves = [image]
    for _ in range(octaves - 1):
        img_octaves.append(nd.zoom(img_octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(img_octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(img_octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr, neuron, offset=offset)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


def get_model(modeltype, vgg_layer, resnet_bnck, resnet_sub_bnck, *args, **kwargs):
    if modeltype == "resnet50_coco":
        network = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        layers = list(network.backbone.body.children())

        # take the 4 layers before into account
        resnet_bnck_id = resnet_bnck + 4

        model = nn.Sequential(*layers[: (resnet_bnck_id + 1)])

        last_module = nn.Sequential()
        for i, module in enumerate(model[resnet_bnck_id]):
            if i > resnet_sub_bnck:
                break
            last_module.add_module(str(i), module)
        model[resnet_bnck_id] = last_module

    if modeltype == "vgg19":
        network = models.vgg19(pretrained=True)
        layers = list(network.backbone.body.children())
        model = nn.Sequential(*layers[: (vgg_layer + 1)])

    if torch.cuda.is_available:
        model = model.cuda()

    return model


def get_filename(*args, **kwargs):
    if kwargs['input_image'] is not None:
        return splitext(basename(kwargs['input_image']))[0] + ".jpg"

    filename = kwargs['modeltype']
    if kwargs['modeltype'] == "resnet50_coco":
        filename += "_resnet_bnk_" + str(kwargs['resnet_bnck']) + "_" + str(kwargs['resnet_sub_bnck'])

    if kwargs['modeltype'] == "vgg19":
        filename += "_vgg-layer_" + str(kwargs['layer'])

    filename += "_neur_" + str(kwargs['neuron']).zfill(4) + "_oct_" + str(kwargs['octaves']) + "_octscale_" + str(
        kwargs['octave_scale']) + "_its_" + str(kwargs['iterations']) + "_offset_" + str(kwargs['offset']) + ".jpg"

    return filename


def get_input(img_initialization, input_image, img_width, img_height, *args, **kwargs):
    if img_initialization == "input_img":
        image = np.array(pil.open(input_image))[:, :, 0:3].astype(np.uint8)

    if img_initialization == "black":
        image = np.zeros((img_height, img_width, 3)).astype(np.uint8)

    return image


def run(*args, **kwargs):
    os.makedirs(kwargs['out_path'], exist_ok=True)
    image = get_input(**kwargs)
    model = get_model(**kwargs)

    dreamed_image = deep_dream(image, model, **kwargs)
    dreamed_image = (dreamed_image * 255).astype(np.uint8).clip(0, 255)
    pil.fromarray(dreamed_image).save(os.path.join(kwargs['out_path'], get_filename(**kwargs)), quality=90)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", default=30, type=int, help="number of gradient ascent steps per octave")
    parser.add_argument("--vgg_layer", default=27, type=int, help="vgg-layer used")
    parser.add_argument("--resnet_bnck", default=2, type=int, help="bottleneck of the resnet50-layer used")
    parser.add_argument("--resnet_sub_bnck", default=0, type=int, help="sub_bottleneck of the resnet50-layer used")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate, the higher the more detail")
    parser.add_argument("--octave_scale", default=1.4, type=float, help="image scale between octaves")
    parser.add_argument("--octaves", default=20, type=int, help="number of octaves, the higher the more recursion")
    parser.add_argument("--neuron", default=478, type=int,
                        help="neuron/channel at which the loss is computed, -1 evaluates all neurons/channels")
    parser.add_argument("--offset", default=0.0001, type=float,
                        help="neuron/channel at which the loss is computed, -1 evaluates all neurons/channels")
    parser.add_argument("--img_width", default=2560, type=int,
                        help="image width, if img_initialization is not input_img. maximum size depends on gpu memory.")
    parser.add_argument("--img_height", default=1440, type=int,
                        help="image height, if img_initialization is not input_img maximum size depends on gpu memory.")
    parser.add_argument("--img_initialization", default="black", choices=['black', 'input_img'],
                        help="initialization of the image, if no input image is provided")
    parser.add_argument("--input_image", type=str, default=None, help="path to input image")
    parser.add_argument("--modeltype", type=str, default="resnet50_coco", choices=["resnet50_coco", "vgg19"],
                        help="network to use for dreaming")
    parser.add_argument("--out_path", default="out", help="output directory")
    parser.add_argument("--from_example", default="", help="load a config from the examples")

    args = vars(parser.parse_args())

    if os.path.isfile(args["from_example"]):
        with open(args["from_example"], 'r') as f:
            args.update(yaml.load(f.read(), Loader=yaml.FullLoader))
        args['input_image'] = basename(args["from_example"])

    run(**args)
