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
from os.path import basename

RESNET_MODELS = {"coco": models.detection.fasterrcnn_resnet50_fpn,
                 "keypoint": models.detection.keypointrcnn_resnet50_fpn,
                 "imagenet": models.resnet50}


class Dreamer:
    def __init__(self):
        self.network = None
        self.model = None
        self.layer = None

    def set_model(self):
        pass

    def get_filename(self, **kwargs):
        filename = "neur_" + str(kwargs['neuron']).zfill(4) + "_oct_" + str(kwargs['octaves']) +\
                   "_octscale_" + str(kwargs['octave_scale']) + "_its_" + str(kwargs['iterations']) +\
                   "_offset_" + str(kwargs['offset']) + "_lr_"+str(kwargs['lr'])+".jpg"
        return filename

    def dream(self, image, iterations, lr, neuron, offset):
        """ Updates the image to maximize outputs for n iterations """
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
        image = Variable(Tensor(image), requires_grad=True)
        for i in range(iterations):
            self.model.zero_grad()
            out = self.model(image)

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

    def deep_dream(self, image, iterations, lr, octave_scale, octaves, neuron, offset, **kwargs):
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
            dreamed_image = self.dream(input_image, iterations, lr, neuron, offset=offset)
            # Extract deep dream details
            detail = dreamed_image - octave_base

        return deprocess(dreamed_image)


class Vgg19Dreamer(Dreamer):

    def __init__(self, type, layer, **kwargs):
        super().__init__()
        self.network = None
        self.type = type
        if self.type == "imagenet":
            self.network = models.vgg19(pretrained=True)
        self.set_model(layer)

    def set_model(self, layer, **kwargs):
        if layer == self.layer:
            return
        self.layer = layer
        layers = list(self.network.features.children())
        model = nn.Sequential(*layers[: (self.layer + 1)])
        if torch.cuda.is_available:
            model = model.cuda()
        self.model = model

    def get_filename(self, *args, **kwargs):
        filename = "vgg19_" + self.type + "_layer_" + str(self.layer)
        filename += "_" + super(Vgg19Dreamer, self).get_filename(**kwargs)
        return filename


class Resnet50Dreamer(Dreamer):

    def __init__(self, type, layer, bottleneck, **kwargs):
        super().__init__()
        self.bottleneck = None
        self.type = type
        self.network = RESNET_MODELS[self.type](pretrained=True)
        self.set_model(layer, bottleneck)

    def set_model(self, layer, bottleneck, **kwargs):
        if (layer == self.layer) and (bottleneck == self.bottleneck):
            return

        self.layer = layer
        self.bottleneck = bottleneck

        layers = list(self.network.backbone.body.children())
        model = nn.Sequential(*layers[: (self.layer + 1)])
        last_module = nn.Sequential()

        for i, module in enumerate(model[layer]):
            if i > self.bottleneck:
                break
            last_module.add_module(str(i), module)
        model[self.layer] = last_module

        if torch.cuda.is_available:
            model = model.cuda()
        self.model = model

    def get_filename(self, **kwargs):
        filename = "resnet50_" + self.type + "_layer_" + str(self.layer) + "_" + str(self.bottleneck)
        filename += "_" + super(Resnet50Dreamer, self).get_filename(**kwargs)
        return filename


def get_input(img_initialization, input_image, img_width, img_height, **kwargs):
    if img_initialization == "input_img":
        image = np.array(pil.open(input_image))[:, :, 0:3].astype(np.uint8)

    if img_initialization == "black":
        image = np.zeros((img_height, img_width, 3)).astype(np.uint8)

    return image


def run(**kwargs):
    os.makedirs(kwargs['out_path'], exist_ok=True)
    image = get_input(**kwargs)

    dreamer = None
    type = kwargs['modeltype'].split("_")[1]

    if "resnet50" in kwargs['modeltype']:
        dreamer = Resnet50Dreamer(type=type, **kwargs)

    if "vgg19" in kwargs:
        dreamer = Vgg19Dreamer(type=type, **kwargs)

    dreamed_image = dreamer.deep_dream(image, **kwargs)
    dreamed_image = (dreamed_image * 255).astype(np.uint8).clip(0, 255)
    pil.fromarray(dreamed_image).save(os.path.join(kwargs['out_path'], dreamer.get_filename(**kwargs)), quality=90)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", default=20, type=int, help="number of gradient ascent steps per octave")
    parser.add_argument("--layer", default=7, type=int, help="layer used")
    parser.add_argument("--bottleneck", default=1, type=int, help="bottleneck only applicable to resnet50 models")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate, the higher the more detail")
    parser.add_argument("--octave_scale", default=1.4, type=float, help="image scale between octaves")
    parser.add_argument("--octaves", default=20, type=int, help="number of octaves, the higher the more recursion")
    parser.add_argument("--neuron", default=1020, type=int,
                        help="neuron/channel at which the loss is computed, -1 evaluates all neurons/channels")
    parser.add_argument("--offset", default=0.001, type=float,
                        help="neuron/channel at which the loss is computed, -1 evaluates all neurons/channels")
    parser.add_argument("--img_width", default=2560, type=int,
                        help="image width, if img_initialization is not input_img. maximum size depends on gpu memory.")
    parser.add_argument("--img_height", default=1440, type=int,
                        help="image height, if img_initialization is not input_img maximum size depends on gpu memory.")
    parser.add_argument("--img_initialization", default="black", choices=['black', 'input_img'],
                        help="initialization of the image, if no input image is provided")
    parser.add_argument("--input_image", type=str, default=None, help="path to input image")
    parser.add_argument("--modeltype", type=str, default="resnet50_coco", choices=["resnet50_coco",
                                                                                   "resnet50_keypoint",
                                                                                   "resnet50_imagenet",
                                                                                   "vgg19_imagenet"],
                        help="network to use for dreaming")
    parser.add_argument("--out_path", default="out", help="output directory")
    parser.add_argument("--from_example", default="", help="load a config from the examples")

    args = vars(parser.parse_args())

    if os.path.isfile(args["from_example"]):
        with open(args["from_example"], 'r') as f:
            args.update(yaml.load(f.read(), Loader=yaml.FullLoader))
        args['input_image'] = basename(args["from_example"])

    run(**args)
