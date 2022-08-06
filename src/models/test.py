import torch
from vgg import get_vgg
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
# from train import New_VGG16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 2
testdata_path = '/home/neiro/PycharmProjects/drova/data/processed/datatest'

transform_test = transforms.Compose([
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         # это среднее и стандартное отклонение всего датасета (обычно imagenet), на котором обучали большую сеть
                         std=[0.229, 0.224, 0.225]),
])

############################
test_data = datasets.ImageFolder(testdata_path, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

net = get_vgg().to(device)
#############################

pred = []
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        #net = New_VGG16()#.to(device)
        net.load_state_dict(torch.load('/home/neiro/PycharmProjects/drova/src/models/best_model.pt'))
        #net.eval()
        y_pred = net(x)

        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1).tolist()
        pred.extend(y_pred)

pred2 = [3 if x == 2 else x for x in pred]

sample = pd.read_csv("/home/neiro/PycharmProjects/drova/data/sample.csv")
sample['class'] = pred2

files = []

for filename in os.listdir(testdata_path + '/testset'):
    if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png']:
        files.append(filename.split(".")[0])
files.sort()

sample['id'] = files
sample['id'] = sample['id'].astype(int)
sample = sample.sort_values(by=['id'])
sample.to_csv('/home/neiro/PycharmProjects/drova/data/sample73.csv', index=False)
