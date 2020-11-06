import glob
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

path = 'obs_data_novelty_new'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48,84)),
    transforms.Lambda(lambda x: (x > 0.5).float()),
])

for subdir in ['']:
    for idx, file in enumerate(glob.glob1(os.path.join(path, subdir), '*.npz')):
        img_name = os.path.join(path, subdir, file)
        obs = dict(np.load(img_name, allow_pickle=True))
        print(obs['state'].shape, obs['next_states'].shape)
        obs['state'] = transform(obs['state'])
        tmp_nextstates = np.zeros((obs['next_states'].shape[0], obs['next_states'].shape[-1], 48, 84))
        for j in range(len(obs['next_states'])):
            tmp_nextstates[j] = transform(obs['next_states'][j])
        obs['next_states'] = tmp_nextstates
        obs['state'] = np.transpose(obs['state'], (1, 2, 0))
        obs['next_states'] = np.transpose(obs['next_states'], (0, 2, 3, 1))
        print(obs['state'].shape, obs['next_states'].shape)

        os.makedirs(os.path.join(path + '_preproc', subdir), exist_ok=True)
        preprocessed_name = os.path.join(path + '_preproc', subdir, file)
        np.savez_compressed(preprocessed_name, **obs)