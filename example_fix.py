import os
from os.path import join
import yaml

in_dir = 'example_configs'
out_dir = 'example_configs2'

os.makedirs(out_dir, exist_ok=True)

files = os.listdir(in_dir)

for file in files:
    print(file)

    with open(join(in_dir, file), "r") as f:
        entry = yaml.load(f.read(), Loader=yaml.FullLoader)

    entry['layer'] = entry['resnet_bnck'] + 4

    if 'resnet_sub_bnck' in entry:
        entry['bottleneck'] = entry['resnet_sub_bnck']
        del entry['resnet_sub_bnck']

    if 'resnet_subnck' in entry:
        entry['bottleneck'] = entry['resnet_subnck']
        del entry['resnet_subnck']

    del entry['resnet_bnck']

    with open(join(out_dir, file), "w") as f:
        yaml.dump(entry, f, sort_keys=True)