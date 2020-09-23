import utils
import models
from standard_dataloader import get_standard_loader
from meta_dataloader import get_meta_loader
import yaml
from tqdm import tqdm

config = yaml.load(open('configs/zwh_train_classifier_mini.yaml', 'r'), Loader=yaml.FullLoader)
model = models.make(config['model'], **config['model_args'])
fs_model = models.make('meta-baseline', encoder=None)
fs_model.encoder = model.encoder

fs_model.eval()
n_shot = 5
n_way = 5
n_query = 15
n_shots = 5
dataloader_test = get_meta_loader(config['dataset_path'], config['test_dataset'], ways=n_way, shots=n_shots, query_shots=n_query, **config['test_dataset_args'])

for data in tqdm(dataloader_test, desc='fs-' + str(n_shot), leave=False):
    train_inputs, train_targets = data["train"]
    print(train_inputs.shape)
    query_inputs, query_targets = data["test"]
    fs_model(train_inputs.view(train_inputs.shape[0], n_way, n_shots, *train_inputs.shape[-3:]), query_inputs)
    # with torch.no_grad():
    #     logits = fs_model(x_shot, x_query).view(-1, n_way)
    #     acc = utils.compute_acc(logits, label)
    # aves['fsa-' + str(n_shot)].add(acc)