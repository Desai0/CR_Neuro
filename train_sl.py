import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import ClashRoyaleAgent
from observation import StatePreprocessor, CARD_LIST
from unit_channels import CHANNEL_COUNT
import glob
import logging

# Constants for coordinate→grid conversion (shared)
PLAYABLE_LOCAL_Y_MIN = 601
PLAYABLE_LOCAL_Y_MAX = 1023
PLAYABLE_LOCAL_X_MIN = 57
CAPTURE_LEFT = 2674
CAPTURE_TOP = 35
PLAYABLE_W = 654
PLAYABLE_H = 422
PLAYABLE_ROWS = 9   # coarse grid playable rows
PLAYABLE_COLS = 9   # coarse grid playable cols
CELL_W = PLAYABLE_W / PLAYABLE_COLS
CELL_H = PLAYABLE_H / PLAYABLE_ROWS
TOTAL_ROWS = 18  # coarse grid rows
TOTAL_COLS = 9   # coarse grid cols

# Relative weight for position loss in total loss (best from hyperparam search)
POS_LOSS_WEIGHT = 0.5


# Helper: convert global screen coords to flattened grid index (0..647)
def coords_to_pos(global_x, global_y):
    local_x = global_x - CAPTURE_LEFT
    local_y = global_y - CAPTURE_TOP

    rel_x = local_x - PLAYABLE_LOCAL_X_MIN
    dist_from_bottom = PLAYABLE_LOCAL_Y_MAX - local_y
    rows_from_bottom = int(dist_from_bottom / CELL_H)

    gx = int(rel_x / CELL_W)
    gy = (TOTAL_ROWS - 1) - rows_from_bottom

    gx = max(0, min(gx, TOTAL_COLS - 1))
    gy = max(0, min(gy, TOTAL_ROWS - 1))

    return gy * TOTAL_COLS + gx


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class ClashDataset(Dataset):
    def __init__(self, data_dir="data"):
        self.file_list = glob.glob(os.path.join(data_dir, "*.npy"))
        self.samples = []
        self.preproc = StatePreprocessor()
        self.SCALARS_SIZE = self.preproc.SCALARS_SIZE
        self.GRID_ROWS = self.preproc.GRID_ROWS
        self.GRID_COLS = self.preproc.GRID_COLS
        self.GRID_FLAT_SIZE = self.preproc.GRID_FLAT_SIZE
        
        logging.info(f"Loading data from {len(self.file_list)} files...")
        
        stats = {"loaded_files": 0, "total_samples": 0, "errors": 0, "skipped": 0, "converted_legacy": 0}
        expected_total_len = self.SCALARS_SIZE + self.GRID_FLAT_SIZE
        LEGACY_ROWS = 36
        LEGACY_COLS = 18
        LEGACY_GRID_FLAT = CHANNEL_COUNT * LEGACY_ROWS * LEGACY_COLS
        legacy_total_len = self.SCALARS_SIZE + LEGACY_GRID_FLAT
        
        for f in self.file_list:
            try:
                data = np.load(f, allow_pickle=True)
                # data shape: (N_samples, 2) -> (Obs, Action)
                file_samples = 0
                for item in data:
                    obs, action = item
                    # Action format: (slot_index, x, y)
                    if action is None: 
                        continue
                    if len(obs) == expected_total_len:
                        self.samples.append((obs, action))
                    elif len(obs) == legacy_total_len:
                        # Convert legacy 36x18 grid to new 18x9 grid by 2x2 average pooling
                        scalars = obs[:self.SCALARS_SIZE]
                        grid_flat_legacy = obs[self.SCALARS_SIZE:]
                        try:
                            grid_legacy = grid_flat_legacy.reshape(CHANNEL_COUNT, LEGACY_ROWS, LEGACY_COLS)
                            grid_new = grid_legacy.reshape(CHANNEL_COUNT, LEGACY_ROWS//2, 2, LEGACY_COLS//2, 2).mean(axis=(2,4))
                            grid_new_flat = grid_new.flatten()
                            new_obs = np.concatenate([scalars, grid_new_flat]).astype(np.float32)
                            self.samples.append((new_obs, action))
                            stats["converted_legacy"] += 1
                        except Exception as conv_e:
                            stats["skipped"] += 1
                            logging.error(f"Error converting legacy sample in {f}: {conv_e}")
                            continue
                    else:
                        stats["skipped"] += 1
                        continue
                    file_samples += 1
                
                if file_samples > 0:
                    stats["loaded_files"] += 1
                    stats["total_samples"] += file_samples
                    
            except Exception as e:
                logging.error(f"Error loading {f}: {e}")
                stats["errors"] += 1
        
        logging.info(f"Dataset Stats: {stats}")
        if len(self.samples) == 0:
            logging.warning("No valid samples loaded!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, action = self.samples[idx]
        
        # Split vector
        scalars = obs[:self.SCALARS_SIZE]
        grid_flat = obs[self.SCALARS_SIZE:]
        
        # Safety check for shape
        if len(grid_flat) != self.GRID_FLAT_SIZE:
             raise ValueError(f"Grid size mismatch! Expected {self.GRID_FLAT_SIZE}, got {len(grid_flat)}")

        # Reshape grid to (Channels, 18, 9) for CNN
        from unit_channels import CHANNEL_COUNT
        grid = grid_flat.reshape(CHANNEL_COUNT, self.GRID_ROWS, self.GRID_COLS)
        
        # Targets
        slot_target = action[0] # 0-3
        
        # --- Grid Conversion Logic (Mirrors observation.py) ---
        global_x, global_y = action[1], action[2]
        pos_target = coords_to_pos(global_x, global_y)
        
        return {
            'grid': torch.tensor(grid, dtype=torch.float32),
            'scalars': torch.tensor(scalars, dtype=torch.float32),
            'slot_target': torch.tensor(slot_target, dtype=torch.long),
            'pos_target': torch.tensor(pos_target, dtype=torch.long)
        }

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=15, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_accuracy(slot_out, pos_out, slot_target, pos_target):
    """Calculate prediction accuracy for both heads."""
    slot_pred = torch.argmax(slot_out, dim=1)
    pos_pred = torch.argmax(pos_out, dim=1)
    
    slot_acc = (slot_pred == slot_target).float().mean().item()
    pos_acc = (pos_pred == pos_target).float().mean().item()
    
    return slot_acc, pos_acc

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting training on device: {device}")
    
    preproc = StatePreprocessor()
    full_dataset = ClashDataset()
    
    if len(full_dataset) == 0:
        logging.error("Dataset is empty. Aborting training.")
        return
    
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logging.info(f"Dataset split: Train={train_size}, Val={val_size}")
    
    # Regular shuffle (no weighted sampling)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    try:
        model = ClashRoyaleAgent(scalars_size=preproc.SCALARS_SIZE, dropout_rate=0.4).to(device)
        logging.info("Model initialized with 2 conv layers, dropout=0.4.")
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        return
    
    # Optimizer (best from hyperparam search)
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Loss without class weights (best from hyperparam search)
    criterion_slot = nn.CrossEntropyLoss(label_smoothing=0.10)
    criterion_pos = nn.CrossEntropyLoss(label_smoothing=0.10)
    
    # Early stopping (tighter patience to avoid late overfit)
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)
    
    epochs = 500 # Max epochs, early stopping will terminate if needed
    logging.info(f"Starting training loop (max {epochs} epochs)...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        train_slot_acc = 0
        train_pos_acc = 0
        train_batches = 0
        
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
                
                loss = loss_s + POS_LOSS_WEIGHT * loss_p
                loss.backward()
                
                # Gradient Clipping (prevent exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                slot_acc, pos_acc = calculate_accuracy(slot_out, pos_out, target_slot, target_pos)
                train_slot_acc += slot_acc
                train_pos_acc += pos_acc
                train_batches += 1
                
            except Exception as e:
                logging.error(f"Error in training batch: {e}")
                continue
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_train_slot_acc = train_slot_acc / train_batches if train_batches > 0 else 0
        avg_train_pos_acc = train_pos_acc / train_batches if train_batches > 0 else 0
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        val_slot_acc = 0
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
                    
                    loss = loss_s + POS_LOSS_WEIGHT * loss_p
                    
                    val_loss += loss.item()
                    slot_acc, pos_acc = calculate_accuracy(slot_out, pos_out, target_slot, target_pos)
                    val_slot_acc += slot_acc
                    val_pos_acc += pos_acc
                    val_batches += 1
                    
                except Exception as e:
                    logging.error(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_val_slot_acc = val_slot_acc / val_batches if val_batches > 0 else 0
        avg_val_pos_acc = val_pos_acc / val_batches if val_batches > 0 else 0
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train Slot Acc: {avg_train_slot_acc:.3f} | Val Slot Acc: {avg_val_slot_acc:.3f} | "
                f"Train Pos Acc: {avg_train_pos_acc:.3f} | Val Pos Acc: {avg_val_pos_acc:.3f}"
            )
        
        # Save best model (based on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join("models", "clash_model_best.pth")
            torch.save(model.state_dict(), best_path)
            # Use ASCII-only to avoid Windows console encoding issues
            logging.info(f"[BEST] New best model saved (Val Loss: {best_val_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join("checkpoints", f"clash_model_ep{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info(f"[EarlyStopping] Training stopped at epoch {epoch+1}")
            break
    
    # Save final model
    final_path = os.path.join("models", "clash_model_v1.pth")
    torch.save(model.state_dict(), final_path)
    logging.info(f"Training complete. Final model saved to {final_path}")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
