import numpy as np
from numpy.random import randn
from scipy.linalg import toeplitz
from torch.utils.data import Dataset
import torch


class StripesDataset(Dataset):
    def __init__(self):
        self.samples = create_stripes()

    def __len__(self):
        return self.samples[0].shape[0]

    def __getitem__(self, idx):
        sample = (self.samples[0][idx], self.samples[1][idx])
        return sample


def create_stripes(num_images_per_class=100, img_shape=(16, 16), num_classes=4,
                   class_type=('vertical', 'horizontal', 'main_diagonal', 'off_diagonal')):

    if num_classes > 4 or num_classes < 2:
        raise ValueError('stripes data has minimum of 2 classes and maximum of 4 classes')

    data = np.zeros([num_classes * num_images_per_class, 1, *img_shape])

    for i in range(num_images_per_class):

        for j in range(num_classes):
            if class_type[j] == 'vertical':
                data[i + j * num_images_per_class] = np.kron(np.ones([img_shape[0], 1]), randn(1, img_shape[1]))

            elif class_type[j] == 'horizontal':
                data[i + j * num_images_per_class] = np.kron(np.ones([1, img_shape[0]]), randn(img_shape[1], 1))

            elif class_type[j] == 'main_diagonal':
                # diagonal (top left to bottom right)
                data[i + j * num_images_per_class] = toeplitz(randn(img_shape[0]), randn(img_shape[1]))

            elif class_type[j] == 'off_diagonal':
                # diagonal (top right to bottom left)
                data[i + j * num_images_per_class] = np.rot90(toeplitz(randn(img_shape[0]), randn(img_shape[1])),
                                                                    1)

    # data = rescale(data)
    labels = np.kron(np.arange(num_classes), np.ones([num_images_per_class]))

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels).long()


if __name__ == '__main__':
    dataset = StripesDataset()
    print(len(dataset))
    print(dataset[100])
    print(dataset[122:361])
