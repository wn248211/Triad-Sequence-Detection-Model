import torch
import librosa
import librosa.display
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import os.path
from torchvision import transforms
from torch.utils import data
import math
import torch.cuda
from spp_layer import spatial_pyramid_pool

classes = ('major', 'minor', 'diminished')

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# def one_hot_encode(labels):
#     n_labels = len(labels)
#     n_unique_labels = len(np.unique(labels))
#     one_hot_encode = np.zeros((n_labels,n_unique_labels))
#     one_hot_encode[np.arange(n_labels), labels] = 1
#     return one_hot_encode

class TriadNet(torch.nn.Module):

  def __init__(self):
    super(TriadNet, self).__init__()
    self.output_num = [32, 16, 8]

    self.conv1 = torch.nn.Sequential(
      torch.nn.Conv2d(1, 16, 5, padding=1),
      torch.nn.BatchNorm2d(16),
      torch.nn.ReLU(inplace=True),
      torch.nn.MaxPool2d(2, 2),
    )
    self.conv2 = torch.nn.Sequential(
      torch.nn.Conv2d(16, 32, 3),
      torch.nn.BatchNorm2d(32),
      torch.nn.ReLU(inplace=True),
      torch.nn.MaxPool2d(2, 2)
    )
    self.fc1 = torch.nn.Linear(43008, 512)
    self.fc2 = torch.nn.Linear(512, 3)

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    # Flatten
    # print(out.view(out.size(0), -1).size())
    # print(out.size())
    feature = spatial_pyramid_pool(self, out, out.size(0), [int(out.size(2)), int(out.size(3))], self.output_num)
    out = self.fc1(feature)
    out = self.fc2(out)
    return out, feature


class TriadData(data.Dataset):
  
  def __init__(self, root, transforms=None, train=True):
    if train:
      root = os.path.join(root, 'trainset')
    else:
      root = os.path.join(root, 'testset')
    self.root = root
    self.train = train
    self.transforms = transforms
    self.audio_files = []
    i = 0
    for label in classes:
      for audio in os.listdir(os.path.join(root, label)):
        self.audio_files.append([os.path.join(root, label, audio), i])
      i = i + 1

  def __getitem__(self, index):
    data_path, label = self.audio_files[index]
    sample_rate = 44100

    # load and preprocess the data
    data = librosa.load(data_path, sr=44100)[0]
    data = data[22050:157662]

    # feature extraction
    data = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=256, n_fft=2048)
    data = librosa.power_to_db(data)
    # data = librosa.feature.chroma_cqt(data, sr=sample_rate)
    if self.transforms:
      data = self.transforms(data)
    return data, label

  def __len__(self):
    return len(self.audio_files)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transforms = transforms.Compose([
  transforms.ToTensor(), 
  # transforms.Normalize(mean=mean, std=std)
])

train_data = TriadData(root='data', transforms=transforms, train=True)
test_data = TriadData(root='data', transforms=transforms, train=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

# load model
model = TriadNet()
model = torch.load('./cnn_model.pkl')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

# Create optimizer SGD
optimizer = optim.SGD(list(model.fc2.parameters()), lr=0.0005, momentum=0.9)

# Training
for epoch in range(200):
  # num = 0
  for i, data in enumerate(train_loader, 0):
    inputs, label = data

    inputs = inputs.to(device)
    label = label.to(device)

    # zero the gradient buffers
    optimizer.zero_grad()

    output, feature = model(inputs)
    loss = criterion(output, label)
    score = F.softmax(output, dim=1)
    # score_list = score.cpu().detach().numpy().tolist()
    # prediction = score_list[0].index(max(score_list[0]))
    # print(prediction)
    # print(label)
    # if prediction == label.cpu().numpy():
    #   num += 1

    loss.backward()
    optimizer.step()
    running_loss = loss.item()
    print(epoch, ": ", running_loss)
    # print(score)
  # print(num/data_loader.__len__())

# save model
torch.save(model, './cnn_model.pkl')

# Testing
# for epoch in range(1):
#   correct_num = 0
#   for i, data in enumerate(test_loader, 0):
#     inputs, label = data
#
#     inputs = inputs.to(device)
#     label = label.to(device)
#
#     output, feature = model(inputs)
#     print(output)
#     # print(output.size())
#     score = F.softmax(output, dim=1)
#     score_list = score.cpu().detach().numpy().tolist()
#     prediction = score_list[0].index(max(score_list[0]))
#     if prediction == label.cpu().numpy():
#       correct_num += 1
#
#   print(correct_num / test_loader.__len__())



