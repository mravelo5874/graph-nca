from data.generate import generate_line_graph, generate_square_plane_graph, generate_bunny_graph
from torch_geometric.nn import PairNorm
from argparse import Namespace
from typing import Optional
from pool import TrainPool
from egnn import egnn, egc
import numpy as np
import datetime
import torch
import json
import os

class MorphogenicEGNCA(torch.nn.Module):
    def __init__(
        self,
    ):
        super(MorphogenicEGNCA, self).__init__()
        
    def forward(
        self,
    ):
        pass
    
class FixedTargetEGNCA(torch.nn.Module):
    def __init__(
        self,
        args: Namespace,
        new_model: bool = True,
    ):
        super(FixedTargetEGNCA, self).__init__()
        self.args = args
        
        if args.graph == 'line':
            target_coords, target_edges = generate_line_graph(args.size)
        elif args.graph == 'grid':
            target_coords, target_edges = generate_square_plane_graph(args.size)
        elif args.graph == 'bunny':
            target_coords, target_edges = generate_bunny_graph()
        else:
            print (f'[models.py] invalid graph: {args.graph}')
            return
            
        print (f'[models.py] target_coords.shape: {target_coords.shape}, target_edges.shape: {target_edges.shape}')
        
        self.register_buffer('target_coords', target_coords * args.scale)
        self.register_buffer('target_edges', target_edges)
        
        # * setup seed coords/hidden
        if new_model:
            print ('[models.py] training new model -- creating starting seed')
            num_nodes = target_coords.size(0)
            seed_coords = torch.empty(num_nodes, args.coords_dim).normal_(std=0.5)
            
            # * save seed data to file
            path = f'{args.save_to}/{args.model_name}'
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(f'{path}/seed_coords.npy', np.array(seed_coords))
            
        else:
            print ('[models.py] loading pre-trained model -- loading starting seed')
        
        # * load in seed for consistency when evaluating
        seed_coords = np.load(f'{self.args.save_to}/{self.args.model_name}/seed_coords.npy')
        seed_coords = torch.tensor(seed_coords).to(self.args.device)
        
        self.normalise = PairNorm(scale=1.0)
        self.mse = torch.nn.MSELoss(reduction='none')
        self.egnn = egnn([
            egc(args.hidden_dim, 
                args.message_dim,
                has_attention=args.has_attention),
        ]).to(args.device)
        self.pool = TrainPool(
            pool_size=args.pool_size,
            hidden_dim=args.hidden_dim,
            seed_coords=seed_coords,
            target_coords=target_coords,
            loss_func=self.mse,
            reset_at=args.reset_at,
        )
        self.optimizer = torch.optim.Adam(
            self.egnn.parameters(), 
            lr=self.args.lr, 
            betas=(self.args.b1, self.args.b2), 
            weight_decay=self.args.wd
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=1e-5,
        )
        
    def forward(
        self,
        batch_coords: torch.Tensor,
        batch_hidden: torch.Tensor,
        edges: torch.LongTensor,
        verbose: Optional[bool] = False
    ):        
        out_coords, out_hidden = self.egnn(
            coords=batch_coords,
            hidden=batch_hidden,
            edges=edges,
            verbose=verbose,
        )
        out_hidden = self.normalise(out_hidden)
        return out_coords, out_hidden
    
    def save(
        self,
        path: str,
        name: str,
        verbose: Optional[bool] = True
    ):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), f'{path}/{name}.pt')
        with open(f'{path}/args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2, default=lambda o: '<not serializable>')
        if verbose: print (f'[models.py] saved model to: {path}')
    
    def get_random_pool_graph(
        self,
    ):
        index = np.random.randint(self.args.pool_size)
        coords = self.pool.cache['coords'][index]
        edges = self.target_edges.clone()
        steps = self.pool.cache['steps'][index]
        return index, coords, edges, steps
        
    def run_for(
        self,
        num_steps: int,
        collect_all: bool = False
    ):
        seed_coords, seed_hidden = self.pool.seed
        coords = seed_coords.clone().to(self.args.device)
        hidden = seed_hidden.clone().to(self.args.device)
        edges = self.target_edges.to(self.args.device)
        
        coords_collection = []
        if collect_all: coords_collection.append(coords)
    
        with torch.no_grad():
            for _ in range(num_steps):
                coords, hidden = self.forward(coords, hidden, edges, verbose=False)
                if collect_all: coords_collection.append(coords)

            rand_target_edges, rand_target_edges_lens = self.pool.get_random_edges()
            edge_lens = torch.norm(coords[rand_target_edges[0]] - coords[rand_target_edges[1]], dim=-1)
            loss_per_edge = self.mse(edge_lens, rand_target_edges_lens)
            loss_per_graph = torch.stack([lpe.mean() for lpe in loss_per_edge.chunk(self.args.batch_size)])
            loss = loss_per_graph.mean()
        
        return coords, edges, loss.item(), coords_collection
    
    def train_model(
        self,
        verbose: bool=False,
        view_random_graphs: bool=False,
    ):
        train_start = datetime.datetime.now()
        loss_log = []
        min_avg_loss = 1e100
        best_model_path = None
        epochs = self.args.epochs+1
        
        # * create edges tensor
        num_nodes = self.target_coords.shape[0]
        edges = []
        for i in range(self.args.batch_size):
            batch_edges = self.target_edges.clone().add(i*num_nodes)
            edges.append(batch_edges)
        edges = torch.cat(edges, dim=1).to(self.args.device)
                    
        for i in range(epochs):
            batch_ids, \
            batch_coords, \
            batch_hidden, \
            rand_edges, \
            rand_target_edges_lens = self.pool.get_batch(self.args.batch_size)
                    
            # * run for n steps
            n_steps = np.random.randint(self.args.min_steps, self.args.max_steps+1)
            for _ in range(n_steps):
                batch_coords, batch_hidden = self.forward(batch_coords, batch_hidden, edges.to(self.args.device), verbose=False)
            
            # * calculate loss
            edge_lens = torch.norm(batch_coords[rand_edges[0]] - batch_coords[rand_edges[1]], dim=-1)
            loss_per_edge = self.mse(edge_lens, rand_target_edges_lens)
            loss_per_graph = torch.stack([lpe.mean() for lpe in loss_per_edge.chunk(self.args.batch_size)])
            loss = loss_per_graph.mean()
            
            # * backward pass
            with torch.no_grad():
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(loss)
                self.pool.update_pool(
                    self.args.batch_size, 
                    batch_ids, 
                    batch_coords, 
                    batch_hidden,
                    n_steps,
                )
                
                # * calc loss and average loss
                _loss = loss.item()
                if not torch.isnan(loss) and not torch.isinf(loss) and not torch.isneginf(loss) and loss != 0.0:
                    loss_log.append(_loss)
                else:
                    print (f'[model.py] detected invalid loss value: {loss} -- stopping training')
                    return
                avg_loss = sum(loss_log[-self.args.info_rate:])/float(self.args.info_rate)
                
                if i % self.args.info_rate == 0 and i!= 0:
                    secs = (datetime.datetime.now()-train_start).seconds
                    if secs == 0: secs = 1
                    time = str(datetime.timedelta(seconds=secs))
                    iter_per_sec = float(i)/float(secs)
                    est_time_sec = int((epochs-i)*(1/iter_per_sec))
                    est = str(datetime.timedelta(seconds=est_time_sec))
                    lr = np.round(self.lr_scheduler.get_last_lr()[0], 8)
                    
                    if view_random_graphs:
                        from utils.visualize import create_ploty_figure_multiple, rgba_colors_list
                        from IPython.display import clear_output
                        from plotly.subplots import make_subplots
                        import plotly
                        
                        # clear_output()
                        trgt_coords = self.target_coords
                        index, pred_coords, edges, steps = self.get_random_pool_graph()
                        seed_coords, _ = self.pool.seed
                        seed_color = rgba_colors_list[4]
                        seed_color[3] = 0
                        print ()
                        plotly.offline.init_notebook_mode()
                        fig1 = create_ploty_figure_multiple(
                            graphs=[(trgt_coords, edges),
                                    (pred_coords, edges),
                                    (seed_coords, edges)],
                            coords_color=[rgba_colors_list[0], rgba_colors_list[1], seed_color],
                            edges_color=[rgba_colors_list[0], rgba_colors_list[1], seed_color]
                        )

                        # * show "run for" graph
                        r_coords, _, r_loss, _ = self.run_for(steps)
                        fig2 = create_ploty_figure_multiple(
                            graphs=[(trgt_coords, edges),
                                    (r_coords, edges),
                                    (seed_coords, edges)],
                            coords_color=[rgba_colors_list[0], rgba_colors_list[1], seed_color],
                            edges_color=[rgba_colors_list[0], rgba_colors_list[1], seed_color]
                        )
                        
                        # * combine figures into one
                        fig3 = make_subplots(
                            rows=1, cols=2, 
                            vertical_spacing=0.02, 
                            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                            subplot_titles=[
                                f'pool graph #{index}, steps: {steps}, loss: {_loss}',
                                f'run_for() graph, steps: {steps} loss: {r_loss}'
                            ])
                        for j in fig1.data :
                            fig3.add_trace(j, row=1, col=1)
                        for j in fig2.data :    
                            fig3.add_trace(j, row=1, col=2)
                        plotly.offline.iplot(fig3)
                    
                    # * print out training stats
                    if verbose: print (f'[{i}/{epochs}]\t {np.round(iter_per_sec, 3)}it/s\t time: {time}~{est}\t loss: {np.round(avg_loss, 8)}>{np.round(np.min(loss_log), 8)}\t lr: {lr}')

                # * save if minimun average loss detected
                if avg_loss < min_avg_loss and i > self.args.info_rate+1:
                    min_avg_loss = avg_loss
                    if best_model_path != None:
                        os.remove(best_model_path)
                    best_model_path = '/'.join([self.args.save_to, self.args.model_name]) + f'/best-{i}.pt'
                    print (f'[models.py] detected minimum average loss during training: {np.round(min_avg_loss, 3)} -- saving model to: {best_model_path}')
                    self.save(
                        path='/'.join([self.args.save_to, self.args.model_name]),
                        name=f'best-{i}',
                        verbose=False,
                    )
                    
def test_model_training(
    num_tests: int = 10,
    verbose: bool = False
):
    torch.set_default_dtype(torch.float32)
    from data.generate import generate_bunny_graph
    seed_coords, seed_edges = generate_bunny_graph()
    
    for i in range(num_tests):
        
        pool_size = np.random.randint(4, 512)
    
        args = Namespace()
        args.scale = 1.0
        args.target_coords = seed_coords
        args.target_edges = seed_edges
        args.coords_dim = 3
        args.hidden_dim = np.random.randint(8, 16)
        args.message_dim = np.random.randint(8, 16)
        args.has_attention = True
        args.device = 'cuda'
        args.pool_size = pool_size
        args.batch_size = np.random.randint(1, pool_size // 2)
        args.epochs = np.random.randint(1, 12)
        args.min_steps = np.random.randint(1, 12)
        args.max_steps = np.random.randint(13, 24)
        args.info_rate = 1e100
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
        fixed_target_egnca.train_model(verbose=False)
        
        if (verbose): print(f'[models.py] test {i} complete')
            
    print(f'[models.py] finished running {num_tests} tests -- all succeeded')