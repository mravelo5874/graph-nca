from argparse import Namespace
from pool import train_pool
from utils import create_graph, expand_edge_tensor, print_batch_dict
from graph import graph
import numpy as np
import datetime
import pickle
import torch
import os

class trainer():
    def __init__(
        self,
        args: Namespace,
        model: torch.nn.Module
    ):
        self.args = args
        self.model = model
        self.target_graph = create_graph(args.graph, args.size, args.length)
        coords, edges, _ = self.target_graph.get()
        self.n_nodes = coords.shape[0]
        self.n_edges = edges.shape[1]
        print(f'[trainer.py] target graph: {self.n_nodes} nodes, {self.n_edges} edges')
    
    def runfor(self):
        raise NotImplementedError
        # this method must be implemented by sub-classes!
        
    def train(self):
        raise NotImplementedError
        # this method must be implemented by sub-classes!

class fixed_target_trainer(trainer):
    def __init__(
        self,
        args: Namespace,
        model: torch.nn.Module
    ):
        super(fixed_target_trainer, self).__init__(args, model)
        
        # check if seed data file already exists (loading trained model)
        path = f'{self.args.save_to}/{self.args.file_name}'
        if os.path.exists(f'{path}/seed.pkl'):
            with open(f'{path}/seed.pkl', 'rb') as file:
                reload_seed = pickle.load(file)
                self.seed_graph = reload_seed
            print (f'[trainer.py] loading seed data from file: {path}/seed.pkl')
        
        # generate new seed graph and save to file
        else:
            # create seed hidden state
            seed_hidden = torch.ones([self.n_nodes, args.hidden_dim])
            if args.init_hidden == 'rand':
                seed_hidden = torch.rand([self.n_nodes, args.hidden_dim])
        
            seed_coords = torch.empty([self.n_nodes, 3]).normal_(std=self.args.seed_std)
            _, target_edges, _ = self.target_graph.get()
            self.seed_graph = graph(seed_coords, target_edges, seed_hidden)
            
            # save seed data to file
            if not os.path.exists(path):
                os.makedirs(path)
            with open(f'{path}/seed.pkl', 'wb') as file:
                pickle.dump(self.seed_graph, file)
                
            # reload seed sanity check
            with open(f'{path}/seed.pkl', 'rb') as file:
                reload_seed = pickle.load(file)
            seed_coords, seed_edges, seed_hidden = self.seed_graph.get()
            reload_coords, reload_edges, reload_hidden = reload_seed.get()
            assert torch.equal(seed_coords, reload_coords)
            assert torch.equal(seed_edges, reload_edges)
            assert torch.equal(seed_hidden, reload_hidden)
            print (f'[trainer.py] generated new seed and saved to file: {path}/seed.pkl')
        
    def runfor(
        self,
        n_steps: int,
        collect_graphs: bool = False
    ):
        assert hasattr(self, 'target_graph')
        assert hasattr(self, 'seed_graph')
        assert hasattr(self, 'model')

        seed_coords, edges, seed_hidden = self.seed_graph.get()
        coords = seed_coords.detach().clone().to(self.args.device)
        hidden = seed_hidden.clone().detach().to(self.args.device)
        coords_collection = []
        if collect_graphs: coords_collection.append(coords.detach().clone().cpu())
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                coords, hidden, _ = self.model(coords, hidden, edges)
                if collect_graphs: coords_collection.append(coords.detach().clone().cpu())
        
        mse = torch.nn.MSELoss(reduction='none')
        temp_pool = train_pool(self.args, self.seed_graph, self.target_graph)
        comp_edges, comp_lens = temp_pool.get_comp_edges(self.args.comp_edge_percent)
        pred_edge_len = torch.norm(coords[comp_edges[0]] - coords[comp_edges[1]], dim=-1)
        loss_per_edge = mse(pred_edge_len, comp_lens)
        loss = loss_per_edge.mean().item()
        
        data = dict()
        data['coords'] = coords.clone().detach().cpu()
        data['edges'] = edges.clone().detach().cpu()
        data['loss'] = loss
        if collect_graphs: data['collection'] = coords_collection
        return data
    
    def train(
        self, 
        vebose=False,
        compare_graphs=False,
    ):
        # create training objects
        self.pool = train_pool(self.args, self.seed_graph, self.target_graph)
        mse = torch.nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.start_lr, 
            betas=(self.args.beta1, self.args.beta2), 
            weight_decay=self.args.wd
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=self.args.end_lr,
        )
        
        # expand target edges tensor
        _, target_edges, _ = self.target_graph.get()
        expanded_edges = expand_edge_tensor(
            target_edges,
            self.n_nodes, 
            self.args.batch_size
        )
        
        print (f'(dev) edges:\n{target_edges}')
        print (f'(dev) expanded_edges:\n{expanded_edges}')
        
        # start training regimen
        loss_log = []
        min_avg_loss = 1e100
        best_model_path = None
        train_start = datetime.datetime.now()
        print (f'[trainer.py] starting training')
        for epoch in range(self.args.epochs):
            
            # put model in train mode (might have switched to eval mode)
            self.model.train()
            
            # gather batch data
            batch_data = self.pool.get_batch(self.args.batch_size, self.args.comp_edge_percent)
            batch_coords = batch_data['coords'].to(self.args.device)
            batch_hidden = batch_data['hidden'].to(self.args.device)
            
            # (dev) test single graph vs whole batch training
            graph_0_coords = batch_coords[0:self.n_nodes,].detach().clone().to(self.args.device)
            graph_0_hidden = batch_hidden[0:self.n_nodes,].detach().clone().to(self.args.device)
            _, graph_0_edges, _ = self.target_graph.get()
            
            # run graphs for n steps
            n = np.random.randint(self.args.min_steps, self.args.max_steps)
            print (f'epoch {epoch}, steps: {n}')
            for _ in range(n):
                batch_coords, batch_hidden, batch_collection = self.model(batch_coords, batch_hidden, expanded_edges, True)
                graph_0_coords, graph_0_hidden, graph_collection = self.model(graph_0_coords, graph_0_hidden, graph_0_edges, True)
                
                from utils import compare_collections
                compare_collections(batch_collection, graph_collection, self.n_nodes, self.n_edges)
                
                coords_diff = batch_coords[0:self.n_nodes,] - graph_0_coords
                hidden_diff = batch_hidden[0:self.n_nodes,] - graph_0_hidden
                # print (f'(dev) coords_diff:\n{coords_diff}')
                # print (f'(dev) hidden_diff:\n{hidden_diff}')
                # assert torch.allclose(batch_coords[0:self.n_nodes,], graph_0_coords)
                # assert torch.allclose(batch_hidden[0:self.n_nodes,], graph_0_hidden)
                
            # configure comparison edges / lengths
            comp_lens = batch_data['comp_lens'].repeat([self.args.batch_size]).to(self.args.device)            
            comp_edges = batch_data['comp_edges']
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
            
            # log loss if it is a valid value
            if not torch.isnan(loss) and \
                not torch.isinf(loss) and \
                not torch.isneginf(loss) and \
                loss != 0.0:
                loss_log.append(_loss)
            else:
                print (f'[trainer.py] detected invalid loss value: {loss} -- stopping training')
                return
            avg_loss = sum(loss_log[-self.args.log_rate:])/float(self.args.log_rate)
            
            # update training pool
            data = dict()
            data['ids'] = batch_data['ids']
            data['coords'] = batch_coords
            data['hidden'] = batch_hidden
            self.pool.update(data, n, loss_per_graph.detach().cpu().numpy())
                
            # collect log info
            if epoch % self.args.log_rate == 0 and epoch != 0:
                elapsed_secs = (datetime.datetime.now()-train_start).seconds
                if elapsed_secs == 0: elapsed_secs = 1
                elapsed_time = str(datetime.timedelta(seconds=elapsed_secs))
                iter_per_sec = float(epoch)/float(elapsed_secs)
                est_time_sec = int((self.args.epochs-epoch)*(1/iter_per_sec))
                est_rem_time = str(datetime.timedelta(seconds=est_time_sec))
                lr = np.round(lr_scheduler.get_last_lr()[0], 8)
                
                # (notebook only) compare training graphs
                if compare_graphs:
                    from utils import compare_pool_vs_runfor_graphs
                    compare_pool_vs_runfor_graphs(self)
                
                # print log info             
                print (f'[{epoch}/{self.args.epochs}]\t {np.round(iter_per_sec, 3)}it/s\t time: {elapsed_time}~{est_rem_time}\t loss: {np.round(avg_loss, 8)}>{np.round(np.min(loss_log), 8)}\t lr: {lr}')
            
            # save model if minimun average loss detected
            if avg_loss < min_avg_loss and epoch >= self.args.log_rate:
                min_avg_loss = avg_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = '/'.join([self.args.save_to, self.args.file_name]) + f'/best@{epoch}.pt'
                self.model.save(f'{self.args.save_to}/{self.args.file_name}', f'best@{epoch}', vebose)
                print (f'[trainer.py] detected minimum average loss during training: {np.round(min_avg_loss, 3)} -- saving model to: {best_model_path}')
            
        # save final model
        self.model.save(f'{self.args.save_to}/{self.args.file_name}', f'final@{epoch}', vebose)
        print (f'[trainer.py] training complete -- saving final model to: {self.args.save_to}/{self.args.file_name}/final@{epoch}.pt')
        
        # log final time
        final_secs = (datetime.datetime.now()-train_start).seconds
        if final_secs == 0: final_secs = 1
        final_time = str(datetime.timedelta(seconds=final_secs))
        print (f'[trainer.py] train time: {final_time}')