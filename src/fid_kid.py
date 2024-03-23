import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import random
from torchmetrics.image.kid import KernelInceptionDistance
import os
from PIL import Image
from torchvision import transforms


def image_loader(directory_path, sample_size=100):
    images = []
    filenames = os.listdir(directory_path)
    random.shuffle(filenames)
    for filename in filenames[:sample_size]:
        image = Image.open(os.path.join(directory_path, filename))
        image = image.convert("RGB")
        image = transforms.Resize((128, 128))(image)
        image = np.array(image)
        images.append(image)
    images = np.array(images)

    images = np.transpose(images, (0, 3, 1, 2))
    image_tensor = torch.tensor(images, dtype=torch.uint8)
    print(images.shape)
    return image_tensor


# set the seed of random number generator
random.seed(0)
for filename in os.listdir(
    "/scratch/st-sdena-1/miladyz/mri_classifiers/data/generated"
):
    print(filename)
    data_dir = (
        "/scratch/st-sdena-1/miladyz/mri_classifiers/data/Alzheimer_MRI_4_classes_dataset/"
        + filename
    )
    data1 = image_loader(data_dir, sample_size=100)
    data_dir = "/scratch/st-sdena-1/miladyz/mri_classifiers/data/generated/" + filename
    data2 = image_loader(data_dir, sample_size=1000)

    fid = FrechetInceptionDistance(feature=64)
    fid.update(data1, real=True)
    fid.update(data2, real=False)
    print("FID:", fid.compute())

    kid = KernelInceptionDistance(subset_size=50)

    kid.update(data1, real=True)
    kid.update(data2, real=False)
    kid_mean, kid_std = kid.compute()
    print("KID", (kid_mean, kid_std))
