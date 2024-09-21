from models import FixedTargetEGNCA
from argparse import Namespace
import numpy as np
import torch
import time

torch.set_default_dtype(torch.float32)
pool_size = 256

# * create args
args = Namespace()
args.save_to = 'logs'
args.model_name = f'line-8'
args.scale = 1.0

args.graph = 'line'
args.size = 8

args.coords_dim = 3
args.hidden_dim = 8
args.message_dim = 16
args.has_attention = True
args.device = 'cuda'
args.pool_size = pool_size
args.batch_size = 16
args.epochs = 100
args.min_steps = 12
args.max_steps = 24
args.info_rate = 10
args.save_rate = 1e100
args.lr = 1e-3
args.b1 = 0.9
args.b2 = 0.999
args.wd = 1e-5
args.patience_sch = 500
args.factor_sch = 0.5
args.density_rand_edge = 1.0

# * create and train model
fixed_target_egnca = FixedTargetEGNCA(args)
tik = time.time()
fixed_target_egnca.train(verbose=True)
tok = time.time()
print('training time: %d (s)' % (tok - tik))

# * save model
fixed_target_egnca.save(
    path='/'.join([args.save_to, args.model_name]), 
    name=f'final-{args.epochs}'
)