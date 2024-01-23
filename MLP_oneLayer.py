import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
#%%
# Load your tabular data
path = r"C:\Users\Omnia\Desktop\Phd\DNA_methy\mVal_cv_feat.csv"
df = pd.read_csv(path, index_col=(0))

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#%%
class TabularModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.dropout = Dropout(p=0.6)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, x):
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train(model, X, y):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 200

    for epoch in range(epochs + 1):
        # Training
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        acc = accuracy(out.argmax(dim=1), y)
        loss.backward()
        optimizer.step()

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>6.2f}%')

    return model

def test(model, X, y):
    """Evaluate the model on the test set and print the accuracy score."""
    model.eval()
    out = model(X)
    acc = accuracy(out.argmax(dim=1), y)
    return acc
#%%
# Instantiate the tabular model
tabular_model = TabularModel(input_dim=X_train.shape[1], hidden_dim=8, output_dim=len(y_train.unique()))

# Train the model
train(tabular_model, torch.Tensor(X.values), torch.LongTensor(y.values))

# Test the model
acc = test(tabular_model, torch.Tensor(X_test.values), torch.LongTensor(y_test.values))
print(f'Tabular model test accuracy: {acc*100:.2f}%\n')
