import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

gound_truth_list = []
answer_list = []
total_epoch = 30
Leaning_Rate = 0.001

model_type = "custom"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        self.classfier = nn.Sequential(
            nn.Linear(in_features=774400, out_features=64)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        x = F.log_softmax(x, -1)
        return x

def fit(epoch, model, data_loader, phase='training', volatile=False, is_cuda=True):
    #  loss, optimizer
    if model_type == "custom":
        criterion = F.cross_entropy
        optimizer = optim.Adam(model.parameters(), lr=Leaning_Rate)

    elif model_type == "vgg":
        criterion = F.cross_entropy
        optimizer = optim.SGD(model.parameters(), lr=Leaning_Rate)

    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()

    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data = data.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

        if phase == 'training':
            optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target).cuda() if is_cuda else criterion(output, target)
        running_loss += criterion(output, target).data

        preds = output.data.max(dim=1, keepdim=True)[1]
        gound_truth = target.data
        answer = preds.squeeze()

        a = gound_truth.data.detach().cpu().numpy()
        b = answer.data.detach().cpu().numpy()

        gound_truth_list.append(a)
        answer_list.append(b)

        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct.item() / len(data_loader.dataset)
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

    return loss, accuracy


def training():

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
        print("cuda support")

    IMG_PATH = "Imgs"

    my_transform = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])

    Images = ImageFolder(IMG_PATH, my_transform)

    print("class2idx:{}".format(Images.class_to_idx))
    print("class:{}".format(Images.classes))
    print("len:{}".format(len(Images.classes)))

    train_size = int(len(Images) * 0.7)
    train, val = torch.utils.data.random_split(Images, [train_size, len(Images) - train_size])

    print("len data1:{}".format(len(train)))
    print("len data2:{}".format(len(val)))

    train_data_loader = torch.utils.data.DataLoader(train, batch_size=16, num_workers=4,  pin_memory=True if is_cuda else False, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(val, batch_size=16, num_workers=4, pin_memory=True if is_cuda else False, shuffle=False)

    print("------------- data load finished -------------------------")

    print("-------------- model selection------------------")

    #  model selection
    if model_type == "custom":
        model = Net().to(device)

    elif model_type == "vgg":
        model = models.vgg19(pretrained=True).to(device)

    summary(model, (3, 224, 224), 16)

    graph_epoch = []
    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []

    print("is_cuda:{}".format(is_cuda))

    for epoch in range(1, total_epoch):

        print("-----------training: {} epoch-----------".format(epoch))

        epoch_loss, epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, valid_data_loader, phase='validation')

        graph_epoch.append(epoch)

        a = epoch_loss.detach().cpu().data.item()
        b = epoch_accuracy
        c = val_epoch_loss.detach().cpu().data.numpy()
        d = val_epoch_accuracy

        train_losses.append(a)
        train_accuracy.append(b)
        val_losses.append(c)
        val_accuracy.append(d)

    x_len = np.arange(len(train_losses))
    plt.plot(x_len, train_losses, marker='.', lw =1, c='red', label="train_losses")
    plt.plot(x_len, val_losses, marker='.', lw =1, c='cyan', label="val_losses")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.plot(x_len, val_accuracy, marker='.', lw =1, c='green', label="val_accuracy")
    plt.plot(x_len, train_accuracy, marker='.', lw =1, c='blue', label="train_accuracy")

    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    gound_truth_list_1 = []
    for idx, data in enumerate(gound_truth_list):
        for j in data:
            gound_truth_list_1.append(j)

    print("gound truth list1:{}".format(gound_truth_list_1))

    ans_truth_list_1 = []
    for idx, data in enumerate(answer_list):
        for j in data:
            ans_truth_list_1.append(j)

    print("ans list2:{}".format(ans_truth_list_1))

if __name__ == '__main__':
    training()
