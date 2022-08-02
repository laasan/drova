import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from prepare_datasets import transferAllClassBetweenFolders, prepareNameWithLabels
import shutil


mean = [0.485, 0.456, 0.406] # это среднее и стандартное отклонение всего датасета (обычно imagenet), на котором обучали большую сеть
std = [0.229, 0.224, 0.225]
classLabels = ['0', '1', '3']

transform_train = transforms.Compose([
         transforms.Resize(1024),
         #transforms.RandomRotation(30),
         transforms.RandomHorizontalFlip(),
         transforms.CenterCrop(1024),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean,
                              std=std),
])

root_path = ''
datasetFolderName = root_path+'data'
sourceFiles = []

X = []
Y = []

train_path = datasetFolderName+'/train/'
validation_path = datasetFolderName+'/validation/'


# Organize file names and class labels in X and Y variables
for i in range(len(classLabels)):
    prepareNameWithLabels(classLabels[i], datasetFolderName, X, Y)

X = np.asarray(X)
Y = np.asarray(Y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, loss_fn, optimizer, train_loader, val_loader, n_epoch=10):
    best_metrics = 0

    for epoch in range(n_epoch):
        print(f'Epoch {epoch + 1}')
        train_dataiter = iter(train_loader)
        for i, batch in enumerate(tqdm(train_dataiter)):
            X_batch, y_batch = batch

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = net(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            metrics = []
            for batch in val_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_pred = net(x)

                y_true = y.detach().cpu().numpy()
                y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
                f1_batch = f1_score(y_true, y_pred, average='macro')
                metrics.append(f1_batch)

            metrics = np.mean(np.array(metrics))
            # если стало лучше - сохраняем на диск и обновляем лучшую метрику
            if metrics > best_metrics:
                print('New best model with test f1 macro:', metrics)
                torch.save(net.state_dict(), './best_model.pt')
                best_metrics = metrics
            if metrics == 1.0:
                break

    return net


vgg16 = models.vgg16(pretrained=True)

vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children()))[:-1]


class New_VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = vgg16
        # for param in self.vgg16.features.parameters():
        #     param.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(4096, 100),
                                nn.Linear(100, 3))

    def forward(self, x):
        x = self.vgg16(x)
        x = self.fc(x)
        return x


net = New_VGG16().to(device)

lr = 5e-4
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# ===============Stratified K-Fold======================
skf = StratifiedKFold(n_splits=3, shuffle=True)
skf.get_n_splits(X, Y)
foldNum = 0
for train_index, val_index in skf.split(X, Y):
    # First cut all images from validation to train (if any exists)
    transferAllClassBetweenFolders('validation', 'train', 1.0, datasetFolderName)
    foldNum += 1
    print("Results for fold", foldNum)
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    # Move validation images of this fold from train folder to the validation folder
    for eachIndex in range(len(X_val)):
        classLabel = ''
        for i in range(len(classLabels)):
            if Y_val[eachIndex] == i:
                classLabel = classLabels[i]
        # Then, copy the validation images to the validation folder
        shutil.move(datasetFolderName + '/train/' + classLabel + '/' + X_val[eachIndex],
                    datasetFolderName + '/validation/' + classLabel + '/' + X_val[eachIndex])

    train_dataset = datasets.ImageFolder(datasetFolderName + '/train/', transform=transform_train)
    val_dataset = datasets.ImageFolder(datasetFolderName + '/validation/', transform=transform_train)

    batch_size = 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    image_datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = ['0', '1', '3']

    net = train(net, loss_fn, optimizer, train_loader, val_loader, n_epoch=5)

