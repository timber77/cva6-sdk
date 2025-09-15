# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Jonas Martin   <martinjo@student.ethz.ch>

# Inference script for demo purposes
print("Starting python imports")
import os
import sys
import time
import argparse

from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter


# Default paths
DEFAULT_MODEL = "./model/resnet18.tflite"
DEFAULT_INPUT_IMG = "./data/dog.bmp"

CLASS_LABELS = "./data/imagenet_classes.txt"



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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run ResNet18 inference on an image')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                        help=f'Path to the TFLite model file (default: {DEFAULT_MODEL})')
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_IMG,
                        help=f'Path to the input image file (default: {DEFAULT_INPUT_IMG})')
    return parser.parse_args()

def main():
    args = parse_arguments()
    MODEL = args.model
    INPUT_IMG = args.input
    
    print(f"Loading: {MODEL}")
    interpreter = Interpreter(MODEL)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Preparing input: {INPUT_IMG}")
    # Load and resize the image
    image = Image.open(INPUT_IMG).convert('RGB')
    image = image.resize((224, 224))

    # Convert to numpy, normalize and reorder axes
    input_data = np.asarray(image, dtype=np.float32) / 255.0  # Shape: [224, 224, 3]
    input_data = np.transpose(input_data, (2, 0, 1))  # Shape: [3, 224, 224]
    input_data = np.expand_dims(input_data, axis=0)  # Shape: [1, 3, 224, 224]

    duration = -1
    interpreter.set_tensor(input_details[0]['index'], input_data)
    print("Invoking tflite interpreter")
    start_time = time.perf_counter()
    interpreter.invoke()
    duration = time.perf_counter() - start_time
    print("Done")
    print(f"Inference time: {duration} seconds")
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print_top5_labels(output_data)


if __name__ == "__main__":
    main()