#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch
import os
"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  python3 detectron2_code/convert-torchvision-to-d2_v2.py detectron2_code/pretrain/WideResNetv2-52-10/epoch400_acc72.82/model_best.pth
  ./convert-torchvision-to-d2.py r50.pth r50.pkl
  # Then, use r50.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"
  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""
def weight_torch_to_detectron2(weight_root):
    obj = torch.load(weight_root, map_location="cpu")['model_state_dict']
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        """
        if "layer" not in k:
            k = "stem." + k
        """
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        """
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        """
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(os.path.join(os.path.dirname(weight_root),os.path.basename(weight_root).replace('pth','pkl')), "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
def weight_torchvision_to_detectron2(weight_root):
    obj = torch.load(weight_root, map_location="cpu")

    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(os.path.join(os.path.dirname(weight_root),os.path.basename(weight_root).replace('pth','pkl')), "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
def weight_torchvision_to_detectron2_dict(obj):
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        """
        if "layer" not in k:
            k = "stem." + k
        """
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}
    return res
def weight_torchvision_to_detectron2_dict_save(save_root, obj):
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        """
        if "layer" not in k:
            k = "stem." + k
        """
        """
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
            
        
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        """
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(os.path.join(os.path.dirname(save_root),os.path.basename(save_root).replace('pth','pkl')), "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
def weight_torchvision_to_detectron2_all_change_dict_save(save_root, obj):
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(os.path.join(os.path.dirname(save_root),os.path.basename(save_root).replace('pth','pkl')), "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
def swinT_weight_to_dt2(save_root):
    source_weights = torch.load(save_root)["model"]
    converted_weights = {}
    keys = list(source_weights.keys())
    
    prefix = 'backbone.bottom_up.'
    for key in keys:
        converted_weights[prefix+key] = source_weights[key]
    torch.save(converted_weights, os.path.join(os.path.dirname(save_root),os.path.basename(save_root).replace('.pth','_dt2.pth')))
    """
    with open(os.path.join(os.path.dirname(save_root),os.path.basename(save_root).replace('.pth','_dt2.pth')), "wb") as f:
        pkl.dump(converted_weights, f)
    if converted_weights:
        print("Unconverted keys:", converted_weights.keys())
    """
"""
save_root =  './detectron2_code/pretrain/swin_tiny_patch4_window7_224.pth'
source_weights = torch.load(save_root)["model"]
print(source_weights)
exit
converted_weights = {}
keys = list(source_weights.keys())

prefix = 'backbone.bottom_up.'
for key in keys:
    converted_weights[prefix+key] = source_weights[key]

"""