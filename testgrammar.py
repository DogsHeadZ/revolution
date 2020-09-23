def get_standard_loader(batch_size, split='ss', shuffle=True, num_workers=0, **kwargs):
    print(batch_size)
    print(split)
    print(shuffle)
    print(kwargs)

kwargs = {'split': 'train',
          'batch_size': 64,
          'augment': 'aug1'}

get_standard_loader(**kwargs)
