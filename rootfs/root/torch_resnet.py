import os
os.environ["OPENBLAS_USE_HERO"] = "" # Unset here because i dont want any calls during import numpy to openblas/ hero inits
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

import perf

DTYPE = torch.float64
torch.set_num_threads(1)

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet34", pretrained=True)
model = model.double()
model.eval()


# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

def hero_init():
    # Initialize hero
    print("Initializing hero")
    os.environ["OPENBLAS_USE_HERO"] = "1"
    perf.openblas_hero_init()
    print("Done")

from PIL import Image
# from torchvision import transforms
input_image = Image.open(filename)

input_image = input_image.resize((256, 256), Image.BILINEAR)

# Center crop to 224x224
width, height = input_image.size
left = (width - 224) // 2
top = (height - 224) // 2
right = left + 224
bottom = top + 224
input_image = input_image.crop((left, top, right, bottom))

# Convert to tensor (C, H, W), normalized to [0, 1]
image_np = np.array(input_image).astype(np.float64) / 255.0  # Shape: (H, W, C)
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)   # Shape: (C, H, W)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
normalized_tensor = (image_tensor - mean) / std

# # Step 1: Un-normalize (reverse normalization)
# mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
# std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# unnormalized_tensor = normalized_tensor * std + mean  # Now in [0, 1]

# # Step 2: Clip to [0, 1] to avoid overflow
# clamped_tensor = unnormalized_tensor.clamp(0, 1)

# # Step 3: Convert to uint8 and PIL Image
# image_np = (clamped_tensor * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, C)
# output_image = Image.fromarray(image_np)

# # Step 4: Save as PNG or JPEG
# output_image.save("output_image.png")


# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
input_batch = normalized_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# hero_init()




print("Running model")
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(input_batch)


# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print("Done")
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)


# Read the categories
with open("/scratch/msc25f15/hero-tools/apps/omp/blas/py/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
