import torch 
import torch.nn as nn  
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def showimg(img):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def data_load(batch_size, show = False):
    transform = transforms.Compose([
        transforms.ToTensor(), #[0,255] na [0,1]
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform=transform)
    
    trainload = torch.utils.data.DataLoader(trainset,batch_size = batch_size,shuffle = True, num_workers = 2 )
    testload = torch.utils.data.DataLoader(testset,batch_size = batch_size,shuffle = False, num_workers = 2 )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if show:
        dataiter = iter(trainload)
        images, labels = next(dataiter)
        showimg(torchvision.utils.make_grid(images))
        for j in range(batch_size):
            print(classes[labels[j]], end=' ')

    return ((trainset,trainload), (testset,testload), classes)







class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5) # filtr 5x5 
        self.pool = nn.MaxPool2d(2,2) 
        self.conv2 = nn.Conv2d(6,16,5) # filtr 5x5
        self.fc1 = nn.Linear(16*5*5,120) #16 warstw 5x5 (bo uzyjemy w forward poola)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) # splaszczenie do liniowych(wyjatek batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

def train(trainload,net, crit, optim):
    epochs = 4
    for epoch in range(epochs):
        r_loss = 0 
        for i, data in enumerate(trainload):
            optim.zero_grad() # zerujemy gradienty
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = crit(outputs, labels) # f straty
            loss.backward() # obliczmy grad dla wszystkich parametrow
            optim.step()
            r_loss += loss.item()
            if i%100 == 99:
                print(f'Epoch: {epoch}, batch: {i+1}, loss: {r_loss/100}')
                r_loss = 0
    print('finished')

def test(testload,net):
    corr = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testload:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0) 
            corr += (predicted == labels).sum().item()
    acc = 100 * corr / total
    print(f'Accuracy of the network on the test images: {acc:.2f} %')

def main():
    (trainset,trainload), (testset,testload), classes = data_load(128, show=True)
    net = cnn()
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    net.to(device)
    train(trainload,net,crit,optim)
    test(testload,net)

if __name__ == '__main__':
    main()