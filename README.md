# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Aaron I
### Register Number:212223230002
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 14)
        self.fc4 = nn.Linear(14, 1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
aaron=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(aaron.parameters(), lr=0.001)


def train_model(aaron, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(aaron(X_train), y_train)
        loss.backward()
        optimizer.step()

        aaron.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

<img width="518" height="685" alt="image" src="https://github.com/user-attachments/assets/2f23b422-5191-42c0-87a1-701b3db6591d" />


## OUTPUT


### Training Loss Vs Iteration Plot

<img width="940" height="596" alt="image" src="https://github.com/user-attachments/assets/f21d98c0-305b-487e-b192-3423b8211222" />


### New Sample Data Prediction

<img width="1153" height="171" alt="image" src="https://github.com/user-attachments/assets/946ee4e1-8b33-432b-996a-4636817edd06" />


## RESULT

Thus, a neural network regression model for the given dataset is successfully developed.
