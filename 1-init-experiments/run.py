from data.generate import generate_line_graph
from models import FixedTargetEGNCA
from argparse import Namespace
import torch
import time

torch.set_default_dtype(torch.float32)
coords, edges = generate_line_graph(4)
pool_size = 256

# * create args
args = Namespace()
args.scale = 1.0
args.target_coords = coords
args.target_edges = edges
args.coords_dim = 3
args.hidden_dim = 8
args.message_dim = 16
args.has_attention = True
args.device = 'cuda'
args.pool_size = pool_size
args.batch_size = 16
args.epochs = 10_000
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
print('Training time: %d (s)' % (tok - tik))