import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd

csv_gt = r"C:\Users\Omnia\Desktop\Phd\DNA_methy\dna_cnv.csv"
df_cnv = pd.read_csv(csv_gt, index_col=(0))
df_cnv.set_index('Folder', inplace=True)

# Split the DataFrame into two modalities
X_modality1 = df_cnv.iloc[:, 0:29]
X_modality2 = df_cnv.iloc[:, 29:-1]
y = df_cnv.iloc[:, -1]

# Split the data into training and testing sets
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X_modality1, X_modality2, y, test_size=0.2, random_state=2)

# Combine the two modalities into a single input for the model
X_train_combined = torch.Tensor(pd.concat([X1_train, X2_train], axis=1).values)
X_test_combined = torch.Tensor(pd.concat([X1_test, X2_test], axis=1).values)

# Define the TabularModel to accept two modalities
class TabularModel(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = Linear(input_dim1, hidden_dim)
        self.fc2 = Linear(input_dim2, hidden_dim)
        self.fc3 = Linear(hidden_dim * 2, output_dim)  # Combining both modalities
        self.dropout = Dropout(p=0.25)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, x1, x2):
        x1 = self.dropout(x1)
        x1 = F.elu(self.fc1(x1))

        x2 = self.dropout(x2)
        x2 = F.elu(self.fc2(x2))

        # Concatenate the outputs from both modalities
        x_combined = torch.cat((x1, x2), dim=1)

        x_combined = self.dropout(x_combined)
        x_combined = self.fc3(x_combined)

        return F.log_softmax(x_combined, dim=1)

# Instantiate the tabular model
tabular_model = TabularModel(input_dim1=X1_train.shape[1], input_dim2=X2_train.shape[1], hidden_dim=1000, output_dim=len(y_train.unique()))

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train(model, X1, X2, y):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 100

    for epoch in range(epochs + 1):
        # Training
        optimizer.zero_grad()
        out = model(X1, X2)  # Pass both modalities to the model
        loss = criterion(out, y)
        acc = accuracy(out.argmax(dim=1), y)
        loss.backward()
        optimizer.step()

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>6.2f}%')

    return model

def test(model, X1, X2, y):
    """Evaluate the model on the test set and print the accuracy score."""
    model.eval()
    out = model(X1, X2)  # Pass both modalities to the model
    acc = accuracy(out.argmax(dim=1), y)
    return acc

# Train the model
train(tabular_model, torch.Tensor(X1_train.values), torch.Tensor(X2_train.values), torch.LongTensor(y_train.values))

# Test the model
acc = test(tabular_model, torch.Tensor(X1_test.values), torch.Tensor(X2_test.values), torch.LongTensor(y_test.values))
print(f'Tabular model test accuracy: {acc*100:.2f}%\n')
