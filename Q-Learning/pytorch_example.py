# pip install torch torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128) # 1st layer to 128 outputs
        self.fc2 = nn.Linear(128, 256) # take 128 outputs and out 256
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        self.loss = nn.CrossEntropyLoss() #nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # send to gpu or cpu
        self.to(self.device) 


    def forward(self, data):
        """ pytorch does backward propagation, we need to implement forward propagation """ 
        layer1 = F.sigmoid(self.fc1(data)) 
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)
        return layer3
    
    def learn(self, data, labels):
        """ take in the labels the data belongs to """
        self.optimizer.zero_grad() # zero the gradient at the start of learning loop
        data = T.tensor(data).to(self.device) # convert tensor to the device applicable type
        labels = T.tensor(labels).to(self.device)
        # T.Tensor vs T.tensor (preserve the input data type)

        predictions = self.forward(data)

        cost = self.loss(predictions, labels)

        cost.backward()
        self.optimizer.step()