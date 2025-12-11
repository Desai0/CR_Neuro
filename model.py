import torch
import torch.nn as nn
import torch.nn.functional as F

from unit_channels import CHANNEL_COUNT

class ClashRoyaleAgent(nn.Module):
    def __init__(self, grid_shape=(CHANNEL_COUNT, 18, 9), n_cards=5, n_actions_slots=4, scalars_size=11, dropout_rate=0.3):
        super(ClashRoyaleAgent, self).__init__()
        
        # 1. Visual Processing (The Grid) with BatchNorm
        # Input: [Batch, CHANNEL_COUNT, 18, 9]
        self.conv1 = nn.Conv2d(CHANNEL_COUNT, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Calculate flattened size after convs 
        # Input: 18x9
        # Conv1 -> 18x9
        # MaxPool(2) -> 9x4 (floor)
        # Conv2 -> 9x4
        # Output channels: 32
        self.flatten_size = 32 * 9 * 4 # 1152
        
        # 2. Scalar Processing (Elixir, Cards in hand features)
        self.fc_scalars = nn.Linear(scalars_size, 32)
        self.dropout_scalars = nn.Dropout(dropout_rate)
        
        # 3. Fusion & Decision
        self.fc_common = nn.Linear(self.flatten_size + 32, 512)
        self.dropout_common = nn.Dropout(dropout_rate)
        
        # --- Output Heads ---
        
        # Head A: Which Card? (Slot 1-4)
        self.head_slot = nn.Linear(512, n_actions_slots)
        
        # Head B: Where to place? (Grid X * Grid Y flat)
        # Output is a probability map over the 9x18 grid (162 classes)
        self.head_pos = nn.Linear(512, 18 * 9)

    def forward(self, grid, scalars):
        # Grid processing with BatchNorm
        x = F.relu(self.bn1(self.conv1(grid)))
        x = F.max_pool2d(x, 2) # Downsample 18x9 -> 9x4
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        
        # Scalar processing with dropout
        y = F.relu(self.fc_scalars(scalars))
        y = self.dropout_scalars(y)
        
        # Fusion
        combined = torch.cat((x, y), dim=1)
        z = F.relu(self.fc_common(combined))
        z = self.dropout_common(z)
        
        slot_logits = self.head_slot(z)
        pos_logits = self.head_pos(z)
        
        return slot_logits, pos_logits

# NOTE: This is a template. We need to align strictly with observation.py output.

