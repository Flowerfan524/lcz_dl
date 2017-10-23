import numpy as np
from torch.utils.data import Dataset


class LCZDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_file, label_file, transform=None):
        """
        Args:
            image_file (string): Path to the numpy file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.lcz_images = np.load(image_file)
        self.label = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image = self.lcz_images[idx]
        label = self.label[idx].squeeze()
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
