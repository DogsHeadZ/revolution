from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from meta_dataloader import get_meta_loader

data_path = 'few_data/'

dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_train=True, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=1, num_workers=4)

# for i, batch in enumerate(dataloader):
#     train_inputs, train_targets = batch["train"]
#     print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
#     print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)
#
#     test_inputs, test_targets = batch["test"]
#     print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
#     print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)
#
#     print(train_targets)
#     if i > 5:
#         break

metadataloader = get_meta_loader(data_path, 'miniimagenet', ways=5, shots=5, query_shots=15,
                                 batch_size=1, split='train')

for i, batch in enumerate(metadataloader):
    train_inputs, train_targets = batch["train"]
    print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
    print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

    test_inputs, test_targets = batch["test"]
    print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
    print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)

    print(train_targets)
    if i > 5:
        break