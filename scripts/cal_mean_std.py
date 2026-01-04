'''
Description: script to calculate mean and std accuracy for each config
Author: jackieysong
Email: jackieysong@tencent.com
Date: 2021-10-14 16:43:06
Acknowledgements: 
'''

import os
import json
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Calculate mean and std accuracy for each config')
parser.add_argument('--config', '-c', type=str, default='basic', help='Using config')
parser.add_argument('--model_name', '-m', type=str, default='CTRGCN', help='Using model')
args, _ = parser.parse_known_args()

root_dir = f'./workdir/{args.config}_{args.model_name}_bone'
if os.path.exists(root_dir):
    models = os.listdir(root_dir)
else:
    # Try other possible path formats
    root_dir = f'./workdir/{args.config}/{args.config}_{args.model_name}_bone'
    if os.path.exists(root_dir):
        models = os.listdir(root_dir)
    else:
        print(f"Directory not found: {root_dir}")
        models = []

acc_dict = dict()
for model in models:
    model_path = os.path.join(root_dir, model)
    acc_path = os.path.join(model_path, 'reco_results.json')
    if not os.path.exists(acc_path):
        continue

    with open(acc_path, 'r') as f:
        acc = json.load(f)
        acc_top1 = float(acc.get('acc_top1'))
        acc_dict[model] = acc_top1

if acc_dict:
    accs = np.array(list(acc_dict.values()))
    acc_max = accs.max()
    acc_mean = accs.mean()
    acc_std = accs.std()
    print(args.config, args.model_name, 'bone', len(accs), f'{acc_max:.2%}', f'{acc_mean:.2%}', f'{acc_std:.2%}')
    print(sorted(acc_dict.items(), key=lambda v: (v[1], v[0]), reverse=True))
else:
    print("No valid accuracy results found.")