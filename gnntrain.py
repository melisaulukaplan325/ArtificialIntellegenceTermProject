import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import numpy as np
import time

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Data Loading and Preprocessing
# MoleculeNet provides pre-featurized molecular datasets, which simplifies the pipeline.
# We download and process the Tox21 dataset.
print("Loading Tox21 dataset (this may take a few minutes)...")
dataset = MoleculeNet(root='./data/Tox21', name='Tox21')
print(f"Dataset loaded: {dataset}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features (per atom): {dataset.num_features}")
# Tox21 is a multi-task dataset with 12 classification targets (e.g., NR-AR, ER-LBD)
NUM_TASKS = dataset.num_classes
print(f"Number of tasks (labels): {NUM_TASKS}")

# Shuffle and split the dataset (standard 80/10/10 split)
torch.manual_seed(42)
dataset = dataset.shuffle()
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

# Create DataLoaders for batch processing
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Definition: GCN for Multi-Task Graph Classification
class GNNClassifier(nn.Module):
    def __init__(self, num_features, hidden_channels, num_tasks):
        super(GNNClassifier, self).__init__()
        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # Final linear layer for 12 tasks (graph-level prediction)
        # 12 output channels, one for each toxicity assay.
        self.lin = nn.Linear(hidden_channels, num_tasks)

    def forward(self, data):
        # x: Node feature matrix, edge_index: Graph connectivity matrix
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # 1. Message Passing (GCN layers)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # 2. Global Pooling (Readout)
        # Reduces node embeddings (x) for each graph in the batch into a single vector.
        # global_mean_pool aggregates features based on the 'batch' index.
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Classification Head
        x = self.lin(x) # [batch_size, num_tasks]
        return x

# Instantiate the model
model = GNNClassifier(
    num_features=dataset.num_features,
    hidden_channels=64,
    num_tasks=NUM_TASKS
).to(device)

print(f"\nModel Architecture:\n{model}")

# 4. Training Utilities
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Multi-task loss function: Binary Cross Entropy with Logits
# The 'reduction="none"' is CRUCIAL for handling missing labels (NaNs).
criterion = nn.BCEWithLogitsLoss(reduction='none')

def train():
    model.train()
    total_loss = 0
    batches_processed = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data)

        # Get ground truth labels
        y = data.y.float()

        # --- Handle Missing Labels (NaNs) ---
        # 1. Identify valid labels (not NaN)
        is_labeled = ~torch.isnan(y)

        # 2. Calculate loss only for labeled targets
        loss_matrix = criterion(out, y)
        labeled_loss = loss_matrix[is_labeled]

        # 3. Robust Mean Calculation (Fix for NaN error)
        if labeled_loss.numel() > 0:
            loss = labeled_loss.mean()

            # Backpropagation
            loss.backward()

            # --- Gradient Clipping (Fix for NaN error) ---
            # Prevents exploding gradients which cause NaN loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            batches_processed += 1
        else:
            # Skip this batch if it has no valid labels
            pass

    if batches_processed == 0:
        return 0.0

    return total_loss / batches_processed

# 5. Evaluation Function
def evaluate(loader):
    model.eval()
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # Get raw logits
            out = model(data)
            # Apply sigmoid to get probabilities for ROC-AUC
            prob = torch.sigmoid(out)

        # Append to lists
        y_true.append(data.y.cpu().numpy())
        y_pred.append(prob.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Check for NaNs in predictions (sign of model collapse)
    if np.isnan(y_pred).any():
        return 0.0, []

    # Calculate AUC for each of the 12 tasks
    task_aucs = []

    # Iterate through all 12 tasks
    for i in range(NUM_TASKS):
        # Extract true and predicted labels for the current task
        y_true_task = y_true[:, i]
        y_pred_task = y_pred[:, i]

        # Filter out missing labels (NaNs)
        is_labeled = ~np.isnan(y_true_task)

        if np.sum(is_labeled) >= 2 and len(np.unique(y_true_task[is_labeled])) == 2:
             # Calculate ROC-AUC only if there are at least 2 samples and 2 classes
            try:
                auc = roc_auc_score(y_true_task[is_labeled], y_pred_task[is_labeled])
                task_aucs.append(auc)
            except ValueError:
                continue

    # Fix for "Mean of empty slice" error
    if len(task_aucs) == 0:
        return 0.0, []

    return np.mean(task_aucs), task_aucs

# 6. Training Loop Execution
NUM_EPOCHS = 50
best_val_auc = 0.0

print("\nStarting Training...")
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train()
    val_auc, _ = evaluate(val_loader)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        # Optional: Save the best model weights
        # torch.save(model.state_dict(), 'best_tox21_gnn_model.pt')

    print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}')

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# 7. Final Evaluation on Test Set
print("\nEvaluating on Test Set...")
test_auc, task_aucs = evaluate(test_loader)
print(f"Final Test ROC-AUC Score (Mean across 12 tasks): {test_auc:.4f}")

# Optional: Print AUC for each task (requires dataset knowledge)
if len(task_aucs) > 0:
    task_names = dataset.task_names
    print("\nIndividual Task ROC-AUC Scores:")
    # Note: The task_aucs list might correspond to a subset of tasks if some were skipped
    # For simplicity in this basic script, we just print what we calculated.
    for i, auc in enumerate(task_aucs):
        print(f"  Task {i+1}: {auc:.4f}")
else:
    print("Could not calculate AUC for any task (possible NaN predictions or insufficient data).")