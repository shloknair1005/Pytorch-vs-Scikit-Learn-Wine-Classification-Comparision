import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
from models.pytorch_model import Model

torch.manual_seed(41)

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
print("-"*100)
print(df.head())
print("-"*100, "\n \n \n")

X = df.drop("target", axis=1).values
y = df["target"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    if i % 10 == 0:
        print(f"Epoch: {i} and loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig("../visualizations/training_loss.png")
plt.close()
plt.show()

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

print(loss)

correct = 0
with torch.no_grad():
  for i, data in enumerate(X_test):
    y_val = model.forward(data)

    # Will tell us what type of flower class our network thinks it is
    print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

    # Correct or not
    if y_val.argmax().item() == y_test[i]:
      correct +=1

print(f'We got {correct} correct!')

with torch.no_grad():
    y_eval = model.forward(X_test)
    test_loss = criterion(y_eval, y_test)
    preds = y_eval.argmax(dim=1)
    accuracy = (preds == y_test).float().mean().item()

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

results = {
    "model": "PyTorch Neural Network",
    "accuracy": float(accuracy),
    "test_loss": float(test_loss),
    "epochs": epochs,
    "architecture": "13-9-10-3"
}

with open("../results/pytorch_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Save model
torch.save(model.state_dict(), "../saved_models/pytorch_model.pth")

