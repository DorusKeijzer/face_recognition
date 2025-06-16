import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import GaussianBlur, ColorJitter
from torchvision import transforms
from torchvision.io import decode_image
from typing import Tuple, Any
import numpy as np
import os
import random
import math

jitter = ColorJitter(brightness=0.5, hue=0.1, contrast=0.5, saturation=0.5)
gauss = GaussianBlur(kernel_size=15, sigma=(1, 5))

resize = transforms.Compose([
    transforms.Resize((224, 224)),
])


def add_random_pixel_noise_tensor(img_t, noise_level=0.1):
    """
    Adds Gaussian noise directly to a tensor image.
    img_t: tensor image (C, H, W) expected in [0,1] range
    """
    noise = torch.randn_like(img_t) * random.gauss(0, 0.1)
    noisy_img_t = img_t + noise
    noisy_img_t = torch.clamp(noisy_img_t, 0., 1.)
    return noisy_img_t

def flexible_transform(img_t):
    """
    img_t: tensor image [C, H, W], assumed in [0,1] range
    """
    apply_jitter = random.choice([True, False])
    apply_gauss = random.choice([True, False])
    apply_noise = random.choice([True, False])

    
    pil_img = transforms.ToPILImage()(img_t)

    if apply_jitter and apply_gauss:
        if random.random() < 0.5:
            pil_img = jitter(pil_img)
            pil_img = gauss(pil_img)
        else:
            pil_img = gauss(pil_img)
            pil_img = jitter(pil_img)
    elif apply_jitter:
        pil_img = jitter(pil_img)
    elif apply_gauss:
        pil_img = gauss(pil_img)
    # else no jitter or gauss

    img_t = transforms.ToTensor()(pil_img)

    if apply_noise:
        img_t = add_random_pixel_noise_tensor(img_t)

    return img_t

class TrainFaceDataset(Dataset):
    def __init__(self) -> None:
        self.img_dir = "./data/Humans/"
        self.file_paths = [x for x in os.listdir(self.img_dir) if os.path.splitext(x)[-1] in [".jpg", ".jpeg", ".png"]]
        self.length = len(self.file_paths)
        assert len(self.file_paths) > 0

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Tuple[torch.Tensor, Any| float | torch.Tensor]:
        file_path = os.path.join(self.img_dir, self.file_paths[index])

        original = resize(decode_image(file_path)).float() / 255.0  # Normalize to [0,1]

        original = resize(decode_image(file_path))
        if original.shape[0] == 1:
            original = original.repeat(3,1,1)
        if original.shape[0] == 4:  # has alpha channel
            original = original[:3, :, :]  # drop the alpha channel
        noisy = flexible_transform(original)

        return original, noisy


train_dataset = TrainFaceDataset()
train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_images = next(iter(train_dataloader))
    original, noisy = train_images  # both [batch_size, 3, 224, 224]

    fig, axs = plt.subplots(2, 10, figsize=(20, 5))
    fig.suptitle('Original (top) vs Noisy (bottom) Images')

    for i in range(10):
        # Original image
        img_orig = original[i].permute(1, 2, 0).cpu().numpy()
        axs[0, i].imshow(img_orig)
        axs[0, i].axis('off')

        # Noisy image
        img_noisy = noisy[i].permute(1, 2, 0).cpu().numpy()
        axs[1, i].imshow(img_noisy)
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()


