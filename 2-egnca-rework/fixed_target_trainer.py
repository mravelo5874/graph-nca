from argparse import Namespace
from pool import train_pool
from utils import create_graph, expand_edge_tensor
from graph import graph
import numpy as np
import datetime
import pickle
import torch
import os

class fixed_target_trainer():
    def __init__(
        self,
        args: Namespace,
        model: torch.nn.Module
    ):
        self.args = args
        self.model = model
        
        # create target graph
        self.target_graph = create_graph(args.graph, args.size, args.length)
        target_coords, _ = self.target_graph.get()
        self.n_nodes = target_coords.shape[0]
        
    def train(self, vebose=False):
        
        # generate seed graph
        _, target_edges = self.target_graph.get()
        seed_coords = torch.empty([self.n_nodes, 3]).normal_(std=self.args.seed_std)
        self.seed_graph = graph(seed_coords, target_edges)
        
        # save seed data to file
        path = f'{self.args.save_to}/{self.args.file_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/seed.pkl', 'wb') as file:
            pickle.dump(self.seed_graph, file)
            
        # reload seed sanity check
        with open(f'{path}/seed.pkl', 'rb') as file:
            reload_seed = pickle.load(file)
        seed_coords, seed_hidden = self.seed_graph.get()
        reload_coords, reload_hidden = reload_seed.get()
        assert torch.equal(seed_coords, reload_coords)
        assert torch.equal(seed_hidden, reload_hidden)
        
        # create training objects
        pool = train_pool(self.args, self.seed_graph, self.target_graph)
        mse = torch.nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.start_lr, 
            betas=(self.args.beta1, self.args.beta2), 
            weight_decay=self.args.wd
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=self.args.end_lr,
        )
        
        # expand target edges tensor
        expanded_edges = expand_edge_tensor(
            target_edges,
            self.n_nodes, 
            self.args.batch_size
        )
        
        # start training regimen
        train_start = datetime.datetime.now()
        for epoch in range(self.args.epochs):
            
            # gather batch data
            batch_data = pool.get_batch(self.args.batch_size)
            batch_coords = torch.tensor(batch_data['coords']).to(self.args.device)
            batch_hidden = torch.tensor(batch_data['hidden']).to(self.args.device)
            
            # run graphs for n steps
            n = np.random.randint(self.args.min_steps, self.args.max_steps)
            for _ in range(n):
                batch_coords, batch_hidden = self.model(batch_coords, batch_hidden, expanded_edges)
                
            # configure comparison edges / lengths
            comp_lens = batch_data['comp-lens']
            comp_lens = torch.tensor([comp_lens]*self.args.batch_size).to(self.args.device)
            comp_edges = batch_data['comp-edges']
            comp_edges = expand_edge_tensor(
                comp_edges, 
                self.n_nodes, 
                self.args.batch_size
            )

            # calculate losses
            pred_edge_len = torch.norm(batch_coords[comp_edges[0]] - batch_coords[comp_edges[1]], dim=-1)
            loss_per_edge = mse(pred_edge_len, comp_lens)
            loss_per_graph = torch.stack([lpe.mean() for lpe in loss_per_edge.chunk(self.args.batch_size)])
            loss = loss_per_graph.mean()
            
            # backwards pass
            with torch.no_grad():
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step(loss)
                _loss = loss.item()
            
            # update training pool
            data = dict()
            data['ids'] = batch_data['ids']
            data['coords'] = batch_coords
            data['hidden'] = batch_hidden
            self.pool.update(batch_data, n, loss_per_graph.cpu().numpy())
                
            # log info
            if epoch % self.args.log_rate == 0 and epoch != 0:
                elapsed_secs = (datetime.datetime.now()-train_start).seconds
                if elapsed_secs == 0: elapsed_secs = 1
                elapsed_time = str(datetime.timedelta(seconds=elapsed_secs))
                print (f'[{epoch}/{self.args.epochs}]\t time: {elapsed_time}')
            
            # save model if minimun average loss detected
            self.model.save(f'{self.args.save_to}/{self.args.file_name}', f'best@{epoch}', vebose)
            
            # (notebook only) show training graphs
            
        # save final model
        self.model.save(f'{self.args.save_to}/{self.args.file_name}', f'final@{epoch}', vebose)
        
        # log final time
        final_secs = (datetime.datetime.now()-train_start).seconds
        if final_secs == 0: final_secs = 1
        final_time = str(datetime.timedelta(seconds=final_secs))
        print (f'[fixed_target_trainer.py] final train time: {final_time}')
        
            