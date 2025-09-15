# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Jonas Martin   <martinjo@student.ethz.ch>

# This is intended to be run on your development machine 
# and will convert the Resnet model provided by torchvision 
# into a tflite model that can then be run on CVA6

import ai_edge_torch
import numpy
import torch
import torchvision

DTYPE = torch.float32 # tflite does not support float64
RESNET_SIZE = "34" # Note: You have to manually modify the weights and model to use
OUTFILE = f"./model/resnet{RESNET_SIZE}.tflite"

resnet = torchvision.models.resnet34(torchvision.models.ResNet34_Weights.IMAGENET1K_V1).eval()

sample_inputs = (torch.randn(1, 3, 224, 224, dtype=DTYPE),)
torch_output = resnet(*sample_inputs)

edge_model = ai_edge_torch.convert(resnet.eval(), sample_inputs)

edge_output = edge_model(*sample_inputs)

if (numpy.allclose(
    torch_output.detach().numpy(),
    edge_output,
    atol=1e-5,
    rtol=1e-5,
)):
    print("Inference result with Pytorch and TfLite was within tolerance")
else:
    print("Something wrong with Pytorch --> TfLite")


edge_model.export(OUTFILE)


import model_explorer
model_explorer.visualize(OUTFILE)
