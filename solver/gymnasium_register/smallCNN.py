import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import json
from pathlib import Path

DATA_PATH = Path(__file__).with_name("solved_state_move_pairs.json")
MODEL_PATH = Path(__file__).with_name("small_cnn_policy.pt")
BATCH_SIZE = 64
EPOCH_RUNS = [5]
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 12345

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_obs(self, board):
        mapping = {".":0, "X":1, "O":2, "U":3, "B":4}
        board = [row.split() for row in board]

        threeDTensor = th.zeros((5, 8, 8), dtype=th.float32)

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                channel = mapping[cell]
                threeDTensor[channel][i][j] = 1.0

        return threeDTensor

def main():
    state_move_pairs = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    move_mapping = {"U": 0, "D": 1, "L": 2, "R": 3}

    xs = []
    ys = []
    for pair in state_move_pairs:
        xs.append(model.get_obs(pair["board"]))
        ys.append(move_mapping[pair["move"]])

    X = th.stack(xs)
    y = th.tensor(ys, dtype=th.long)

    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * VALIDATION_FRACTION)
    train_size = len(dataset) - val_size
    split_generator = th.Generator().manual_seed(SPLIT_SEED)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_examples = 0
        top1_correct = 0
        top2_correct = 0
        top3_correct = 0

        with th.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                top3 = th.topk(output, k=3, dim=1).indices
                top1_correct += (top3[:, 0] == batch_y).sum().item()
                top2_correct += (
                    (top3[:, :2] == batch_y.unsqueeze(1)).any(dim=1).sum().item()
                )
                top3_correct += (
                    (top3 == batch_y.unsqueeze(1)).any(dim=1).sum().item()
                )

        return {
            "loss": total_loss / total_examples,
            "top1": top1_correct / total_examples,
            "top2": top2_correct / total_examples,
            "top3": top3_correct / total_examples,
        }

    print(f"Training examples: {train_size}")
    print(f"Validation examples: {val_size}")
    print(f"Epoch runs: {EPOCH_RUNS}", flush=True)

    total_epochs_done = 0
    total_epochs_to_run = sum(EPOCH_RUNS)

    for run_epochs in EPOCH_RUNS:
        for _ in range(run_epochs):
            total_epochs_done += 1
            model.train()
            total_loss = 0.0
            total_examples = 0

            for batch_x, batch_y in train_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

        train_loss = total_loss / total_examples
        val_stats = evaluate()
        print(
            f"after run={run_epochs} epochs "
            f"total_epoch={total_epochs_done}/{total_epochs_to_run} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_top1={val_stats['top1']:.1%} "
            f"val_top2={val_stats['top2']:.1%} "
            f"val_top3={val_stats['top3']:.1%}",
            flush=True,
        )
        checkpoint_path = Path(__file__).with_name(
            f"small_cnn_policy_epoch_{total_epochs_done}.pt"
        )
        th.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}", flush=True)

    th.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
