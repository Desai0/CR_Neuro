import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import ClashRoyaleAgent
from observation import StatePreprocessor
from train_sl import ClashDataset, EarlyStopping, calculate_accuracy, coords_to_pos, TOTAL_ROWS, TOTAL_COLS
import logging
import itertools
from datetime import datetime
import ctypes

# Disable verbose logging during search
logging.basicConfig(level=logging.WARNING)

class GeneticOptimizerWrapper:
    def __init__(self, pop_size=20, mutation_rate=0.1): 
        # Note: pop_size and mutation_rate are now hardcoded in C++, but we keep args for compatibility
        # We must respect the C++ hardcoded POPULATION_SIZE = 20 for buffer sizing!
        self.pop_size = 20 # Sync with C++
        
        dll_name = "optimizer.dll"
        dll_abspath = os.path.abspath(dll_name)
        dll_dir = os.path.dirname(dll_abspath)
        
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_dir)
            
        try:
            self.lib = ctypes.CDLL(dll_abspath)
        except FileNotFoundError:
            self.lib = ctypes.CDLL(dll_name)
        
        # 1. create_optimizer(int num_params)
        self.lib.create_optimizer.argtypes = [ctypes.c_int]
        self.lib.create_optimizer.restype = ctypes.c_void_p
        
        self.lib.destroy_optimizer.argtypes = [ctypes.c_void_p]
        
        # 2. init_population(ctx, min_vals, max_vals)
        self.lib.init_population.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double)
        ]
        
        # 3. get_individual(ctx, index, out_genes)
        self.lib.get_individual.argtypes = [
            ctypes.c_void_p, 
            ctypes.c_int, 
            ctypes.POINTER(ctypes.c_double)
        ]
        
        # 4. set_score(ctx, index, score)
        self.lib.set_score.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        
        # 5. evolve(ctx)
        self.lib.evolve.argtypes = [ctypes.c_void_p]
        
        # Define 6 params ranges
        # [pos_weight, dropout, lr, weight_decay, batch_size, label_smooth]
        self.min_vals = [0.1, 0.0, 1e-4, 1e-5, 16.0, 0.0]
        self.max_vals = [2.0, 0.5, 1e-2, 1e-3, 128.0, 0.2]
        self.n_params = 6
        
        self.obj = self.lib.create_optimizer(self.n_params)
        
        # Init population
        c_min = (ctypes.c_double * self.n_params)(*self.min_vals)
        c_max = (ctypes.c_double * self.n_params)(*self.max_vals)
        self.lib.init_population(self.obj, c_min, c_max)
        
        print(f"[Python] Wrapper initialized C++ optimizer object at {self.obj}")

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'obj'):
            self.lib.destroy_optimizer(self.obj)

    def get_population(self):
        configs = []
        out_buffer = (ctypes.c_double * self.n_params)()
        
        for i in range(self.pop_size):
            self.lib.get_individual(self.obj, i, out_buffer)
            
            config = {
                'pos_loss_weight': float(out_buffer[0]),
                'dropout': float(out_buffer[1]),
                'lr': float(out_buffer[2]),
                'weight_decay': float(out_buffer[3]),
                'batch_size': int(out_buffer[4]),
                'label_smoothing': float(out_buffer[5]),
            }
            configs.append(config)
        return configs

    def evolve(self, scores):
        # Pass scores manually to C++ before evolving
        for i, score in enumerate(scores):
            # Invert score because C++ does MAXimization, and we minimize Loss.
            # But wait, original code: score = val_loss - acc. Lower is better.
            # C++ sorts descending (a > b). So higher is better.
            # So we should pass -score (negative loss).
            self.lib.set_score(self.obj, i, -score)
            
        self.lib.evolve(self.obj)

class HyperparamSearch:
    def __init__(self, n_generations=5, pop_size=6):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.results = []
        self.best_config = None
        self.best_score = float('inf')
        self.optimizer = GeneticOptimizerWrapper(pop_size=pop_size, mutation_rate=0.3)
        
    def train_with_config(self, config, max_epochs=3): # Reduced epochs for demo
        """Train model with given config and return validation metrics."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare data (could be cached, but for safety reloading)
        preproc = StatePreprocessor()
        full_dataset = ClashDataset()
        
        if len(full_dataset) == 0:
            return None
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        
        # Initialize model
        model = ClashRoyaleAgent(scalars_size=preproc.SCALARS_SIZE, dropout_rate=config['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        
        criterion_slot = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        criterion_pos = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        
        best_val_loss = float('inf')
        best_val_pos_acc = 0.0
        
        for epoch in range(max_epochs):
            model.train()
            # Fast training loop
            for batch in train_loader:
                try:
                    grid = batch['grid'].to(device)
                    scalars = batch['scalars'].to(device)
                    target_slot = batch['slot_target'].to(device)
                    target_pos = batch['pos_target'].to(device)
                    
                    optimizer.zero_grad()
                    slot_out, pos_out = model(grid, scalars)
                    
                    loss_s = criterion_slot(slot_out, target_slot)
                    loss_p = criterion_pos(pos_out, target_pos)
                    loss = loss_s + config['pos_loss_weight'] * loss_p
                    
                    loss.backward()
                    optimizer.step()
                except Exception:
                    continue
            
            # Validation
            model.eval()
            val_loss = 0
            val_pos_acc = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        grid = batch['grid'].to(device)
                        scalars = batch['scalars'].to(device)
                        target_slot = batch['slot_target'].to(device)
                        target_pos = batch['pos_target'].to(device)
                        
                        slot_out, pos_out = model(grid, scalars)
                        loss_s = criterion_slot(slot_out, target_slot)
                        loss_p = criterion_pos(pos_out, target_pos)
                        loss = loss_s + config['pos_loss_weight'] * loss_p
                        
                        val_loss += loss.item()
                        _, pos_acc = calculate_accuracy(slot_out, pos_out, target_slot, target_pos)
                        val_pos_acc += pos_acc
                        val_batches += 1
                    except Exception:
                        continue
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            avg_val_pos_acc = val_pos_acc / val_batches if val_batches > 0 else 0.0
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_pos_acc = avg_val_pos_acc
        
        return {
            'val_loss': best_val_loss,
            'val_pos_acc': best_val_pos_acc,
            'epochs_trained': max_epochs
        }
    
    def run_search(self):
        """Run Genetic Search."""
        print("=" * 80)
        print("HYPERPARAMETER SEARCH (GENETIC ALGORITHM)")
        print(f"C++ Backend: Enabled")
        print("=" * 80)
        print(f"Generations: {self.n_generations}")
        print(f"Population: {self.pop_size}")
        print()
        
        for gen in range(self.n_generations):
            print(f"\n--- Generation {gen+1}/{self.n_generations} ---")
            
            # 1. Get population from C++
            population = self.optimizer.get_population()
            scores = []
            
            # 2. Evaluate
            for i, config in enumerate(population):
                print(f"  [Ind {i+1}] Config: LR={config['lr']:.1e}, Bat={config['batch_size']}, Drop={config['dropout']:.1f}")
                
                result = self.train_with_config(config)
                
                score = float('inf')
                if result:
                    score = result['val_loss'] - (result['val_pos_acc'] * 2.0)
                    print(f"     -> Loss: {result['val_loss']:.4f} | Score: {score:.4f}")
                else:
                    print("     -> Failed")
                
                scores.append(score)
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_config = config
                    print("     ✅ NEW GLOBAL BEST!")

            # 3. Evolve using C++
            print(f"  [Evolution] Passing scores to C++ optimizer...")
            self.optimizer.evolve(scores)

        print("\n🏆 Genetic Search Complete!")
        print(f"Best Config: {self.best_config}")
        print(f"Best Score: {self.best_score}")

if __name__ == "__main__":
    searcher = HyperparamSearch(n_generations=3, pop_size=4)
    searcher.run_search()
