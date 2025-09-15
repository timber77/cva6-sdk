# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Jonas Martin   <martinjo@student.ethz.ch>


print("Starting imports")
import os
import time
import ctypes

from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter


MODEL = "./model/resnet18.tflite"
INPUT_IMG = "./data/dog.bmp"
CLASS_LABELS = "./data/imagenet_classes.txt"

RUN_WITH_ACC = True
RUN_WITHOUT_ACC = False

blaslib = ctypes.CDLL("libopenblas_riscv64_generic-r0.3.27.dev.so")

def openblas_hero_init():
    print("Initializing hero")
    blaslib.openblas_hero_init()

def print_top5_labels(output_data):
    with open(CLASS_LABELS) as f:
        labels = [line.strip() for line in f]

        # Get top 5 predictions and their indices
        top5_indices = np.argsort(output_data[0])[::-1][:5]  # sort in descending order
        top5_confidences = output_data[0][top5_indices]      # get corresponding confidence values

        # Print top 5 predicted labels with their confidence scores
        print("Top 5 Predictions:")
        for i in range(5):
            label = labels[top5_indices[i]]
            confidence = top5_confidences[i]
            print(f"{i+1}: {label} (Confidence: {confidence:.4f})")

def main():
    print(f"Loading {MODEL}")
    interpreter = Interpreter(MODEL)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Preparing input image {INPUT_IMG}")
    # Load and resize the image
    image = Image.open(INPUT_IMG).convert('RGB')
    image = image.resize((224, 224))

    # Convert to numpy, normalize and reorder axes
    input_data = np.asarray(image, dtype=np.float32) / 255.0  # Shape: [224, 224, 3]
    input_data = np.transpose(input_data, (2, 0, 1))  # Shape: [3, 224, 224]
    input_data = np.expand_dims(input_data, axis=0)  # Shape: [1, 3, 224, 224]

    print("Input shape:", input_data.shape)
    input_shape = input_details[0]['shape']
    print("Model requires shape:", input_shape)
    duration_acc = -1
    if RUN_WITH_ACC:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        print("Invoke Interpreter with acceleration")
        os.environ["OPENBLAS_USE_HERO"] = "1"
        start_time_acc = time.perf_counter()
        # openblas_hero_init()
        interpreter.invoke()
        duration_acc = time.perf_counter() - start_time_acc
        print("Done")
        print(f"Inference time: {duration_acc} seconds")
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print_top5_labels(output_data)

    duration_no_acc = -1
    if RUN_WITHOUT_ACC:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        print("Invoke Interpreter without acceleration")
        os.environ["OPENBLAS_USE_HERO"] = ""
        start_time_no_acc = time.perf_counter()
        interpreter.invoke()
        duration_no_acc = time.perf_counter() - start_time_no_acc
        print("Done")
        print(f"Inference time: {duration_no_acc} seconds")
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print_top5_labels(output_data)

    if RUN_WITH_ACC and RUN_WITHOUT_ACC:
        print("Speedup:", duration_no_acc/duration_acc)


if __name__ == "__main__":
    main()