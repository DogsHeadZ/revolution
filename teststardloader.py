from standard_dataloader import Standard_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch

def show_landmarks_batch(image, label):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = image, label
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imsave('ddd.jpg', grid.numpy().transpose((1, 2, 0)))

    # for i in range(batch_size):
    #     plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
    #                 landmarks_batch[i, :, 1].numpy(),
    #                 s=10, marker='.', c='r')
    #
    #     plt.title('Batch from dataloader')

dataset_train = Standard_dataset('miniimagenet', 'few_data/','train', transform=transforms.Compose([transforms.Resize(84),
                               transforms.ToTensor()]))
dataloader = DataLoader(dataset_train, batch_size=1202, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

for i, (image, label) in enumerate(dataloader):
    print(image.shape)
    print(label)
    # array = image.numpy().transpose((1, 2, 0))
    show_landmarks_batch(image, label)
    if i >= 0:
        break

