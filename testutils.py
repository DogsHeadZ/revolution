import os
import shutil
import utils

svname = 'train-classifier'
save_path = os.path.join('./save', svname)
print(save_path)
print(save_path.rstrip('/'))
utils.ensure_path(save_path)