# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Jonas Martin   <martinjo@student.ethz.ch>

# A minimal "server" that runs and checks for new input images and performs image classification using a TFLite model.
# The input images must have been preprocessed to be 224x224 pixels and normalized.
# The classification results will be stored in a output file per input image

# The "server" approach is taken to get rid of repeated imports of python modules




print("Starting python imports")
import os
import sys
import time
import argparse


from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter
from pathlib import Path

# Default paths
DEFAULT_MODEL = "./model/resnet18.tflite"
DEFAULT_INPUT_IMG = "./data/dog.bmp"

CLASS_LABELS = "./data/imagenet_classes.txt"

DEFAULT_INPUT_IMG_FOLDER = "./data/images/"
DEFAULT_OUTPUT_FOLDER = "./data/outputs/"

def print_top5_labels(output_data):
    with open(CLASS_LABELS) as f:
        labels = [line.strip() for line in f]

        # Get top 5 predictions and their indices
        top5_indices = np.argsort(output_data[0])[::-1][:5]  # sort in descending order
        top5_confidences = output_data[0][top5_indices]      # get corresponding confidence values

        # Print top 5 predicted labels with their confidence scores
        top5_string = "Top 5 Predictions:\n" + "\n".join([f"{i+1}: {labels[top5_indices[i]]} (Confidence: {top5_confidences[i]:.4f})"
                                  for i in range(5)]) + "\n"
        print(top5_string)
        return top5_string

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run ResNet18 inference on an image')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                        help=f'Path to the TFLite model file (default: {DEFAULT_MODEL})')
    # parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_IMG,
    #                     help=f'Path to the input image file (default: {DEFAULT_INPUT_IMG})')
    parser.add_argument('--input_folder', '-if', type=str, default=DEFAULT_INPUT_IMG_FOLDER,
                        help=f'Path to the input image folder (default: {DEFAULT_INPUT_IMG_FOLDER})')
    parser.add_argument('--output_folder', '-of', type=str, default=DEFAULT_OUTPUT_FOLDER,
                        help=f'Path to the output folder (default: {DEFAULT_OUTPUT_FOLDER})')

    return parser.parse_args()


def get_new_image(folder:Path) -> Path | None:
    """
    Selects a new input image from the input folder and returns its name 
    """
    image_files = [f for f in os.listdir(folder) if f.endswith(('.bmp'))]
    if not image_files:
        print("No new images found.")
        return None
    return Path(os.path.join(folder, image_files[0]))

def load_new_image(file:Path) -> np.array:
    """
    Loads a new image from the given file path, the image shall already have the correct size of 224x224
    """
    image = Image.open(file).convert('RGB')
    input_data = np.asarray(image, dtype=np.float32) / 255.0  # Shape: [224, 224, 3]
    input_data = np.transpose(input_data, (2, 0, 1))  # Shape: [3, 224, 224]
    input_data = np.expand_dims(input_data, axis=0)  # Shape: [1, 3, 224, 224]
    return input_data

def delete_image(file:Path):
    """
    Deletes the specified image file.
    """
    try:
        os.remove(file)
        print(f"Deleted image: {file}")
    except Exception as e:
        print(f"Error deleting image {file}: {e}")

def report_results(output_folder: Path, input_file: Path, output_data: np.array):
    """
    Reports the classification results.
    """
    print(f"Results for {input_file}:")
    top5_string = print_top5_labels(output_data)

    with open(Path(output_folder) / f"{input_file.stem}_results.txt.tmp", "w") as f:
        f.write(top5_string)
        f.write("All Predictions:\n")
        f.write(np.array2string(output_data, max_line_width=10**10))
    os.rename(Path(output_folder) / f"{input_file.stem}_results.txt.tmp",
              Path(output_folder) / f"{input_file.stem}_results.txt")


def run_inference(interpreter, input_details, input_data, output_details):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    print("Invoking interpreter...")
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def server_main_loop(interpreter, input_folder, output_folder):

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        new_image_path:Path = get_new_image(input_folder)
        if new_image_path is None:
            time.sleep(5)  # Wait before checking for new images again
            continue
        if new_image_path.stem == "STOP":
            delete_image(new_image_path)
            break

        print(f"Loading input: {new_image_path}")
        input_data = load_new_image(new_image_path)

        output_data = run_inference(interpreter, input_details, input_data, output_details)

        report_results(output_folder, new_image_path, output_data)
        delete_image(new_image_path)

def main():
    args = parse_arguments()
    MODEL = args.model
    INPUT_IMG_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder

    Path(INPUT_IMG_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print(f"Loading: {MODEL}")
    interpreter = Interpreter(MODEL)
    interpreter.allocate_tensors()

    print("Starting server main loop")
    server_main_loop(interpreter, INPUT_IMG_FOLDER, OUTPUT_FOLDER)
    print("Server stopped.")


if __name__ == "__main__":
    main()