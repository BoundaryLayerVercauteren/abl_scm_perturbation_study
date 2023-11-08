import os
import shutil

import numpy as np

uGs = np.arange(0.6, 3.6, 0.1)

directory_path = 'single_column_model/solution/short_tail/perturbed/pde_theta/neg/simulations/'

for uG in uGs:
    uG = str(np.round(uG, 1))
    sub_dir_path = directory_path + uG.replace('.', '_')

    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

    for file in os.listdir(directory_path):
        if uG in file:
            old_file_path = directory_path + file
            new_file_path = sub_dir_path + '/' + file
            shutil.move(old_file_path, new_file_path)
