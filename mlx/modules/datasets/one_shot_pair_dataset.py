import os
import cv2
import random
import torch
from torch.utils.data import Dataset


class OneShotPairDataset(Dataset):
    """
    A dataset class for one-shot learning using OpenCV.

    Generates (img1, img2, label):
      - label = 1 → same class
      - label = 0 → different classes
    """
    def __init__(self, root_dir, input_size=(105, 105), colored=True, n_pairs_per_class=100):
        self.root_dir = root_dir
        self.input_size = input_size
        self.colored = colored
        self.n_pairs_per_class = n_pairs_per_class

        # Scan dataset directories (character folders)
        self.class_to_images = self._index_images()
        self.classes = list(self.class_to_images.keys())

    def _index_images(self):
        """Map each class to a list of image paths."""
        class_to_images = {}
        for label, subdir in enumerate(sorted(os.listdir(self.root_dir))):
            path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(path):
                continue

            image_files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if len(image_files) >= 2:
                class_to_images[label] = image_files
        return class_to_images

    def __len__(self):
        # Each class generates several random pairs
        return len(self.classes) * self.n_pairs_per_class

    def __getitem__(self, idx):
        same_class = random.random() < 0.5

        if same_class:
            class_label = random.choice(self.classes)
            img_paths = random.sample(self.class_to_images[class_label], 2)
            label = 1
        else:
            c1, c2 = random.sample(self.classes, 2)
            img_paths = [
                random.choice(self.class_to_images[c1]),
                random.choice(self.class_to_images[c2]),
            ]
            label = 0

        img1 = self._load_image(img_paths[0])
        img2 = self._load_image(img_paths[1])

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def _load_image(self, path):
        if self.colored:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img[..., None]  # add channel dim for consistency

        # Resize and normalize
        img = cv2.resize(img, self.input_size)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img
