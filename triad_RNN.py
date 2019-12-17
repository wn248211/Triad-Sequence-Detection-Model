import torch
import librosa
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import os.path
from torchvision import transforms
import torchvision
from torch.utils import data
import torch.cuda
from triad_classification import TriadNet
from torch import nn

max_epoch = 500
lr = 0.001

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class TriadSeq(data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.audio_files = []
        self.cnn_model = torch.load('./cnn_model.pkl')
        # self.cnn_model = TriadNet()
        self.cnn_model = self.cnn_model.to(device)

        with open(os.path.join(root, 'data.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                name = line.split(' ')[0]
                tempo = line.split(' ')[1]
                label = line.split(' ')[2:]
                label = list(map(int, label))
                self.audio_files.append([os.path.join(root, name), tempo, label])

    def __getitem__(self, index):
        data_path, tempo, label = self.audio_files[index]

        # load and preprocess the data
        data, sr = librosa.load(data_path, sr=44100)

        # split audio seq into several pieces
        meter = 60 / int(tempo)
        seq_len = int(data.shape[0] / (meter * sr * 2))
        data_seq = []
        for i in range(2, seq_len):
            data_part = data[int(i * meter * sr * 2):int((i + 1) * meter * sr * 2)]
            data_part = librosa.feature.melspectrogram(data_part, sr=sr, n_mels=256, n_fft=2048)
            data_part = librosa.power_to_db(data_part)
            data_part = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(data_part), 0), 0)
            output, feature = self.cnn_model(data_part.to(device))
            output = output.to("cpu")
            data_part = torch.squeeze(output, 0).detach().numpy()
            data_seq.append(data_part)
        data_seq = np.asarray(data_seq)

        if self.transforms:
            data_seq = self.transforms(data_seq)
        return data_seq, torch.tensor(label)

    def __len__(self):
        return len(self.audio_files)

class triad_rnn(nn.Module):

    # hidden_dim : size of features of hidden state
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(triad_rnn, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, x):
        batch_size = x.size(0)
        seq_size = x.size(1)

        # Initializing hidden state(h0) for first input using method defined below
        hidden_0 = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden_0)

        # flatting
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden


    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden

transforms = transforms.Compose([
  # transforms.ToTensor(),
  # transforms.Normalize(mean=mean, std=std)
])

train_data = TriadSeq(root='data_seq', transforms=transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

rnn_model = triad_rnn(input_size=3, output_size=3, hidden_dim=3, n_layers=1)
rnn_model = rnn_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn_model.parameters(), lr=lr, momentum=0.9)

# Training
for epoch in range(max_epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, label = data
        print(inputs)

        inputs = inputs.to(device)
        label = label.to(device)

        # zero the gradient buffers
        optimizer.zero_grad()

        output, hidden = rnn_model(inputs)
        loss = criterion(output, label.view(-1).long())
        score = F.softmax(output, dim=1)
        score_list = score.cpu().detach().numpy().tolist()
        prediction = score_list[0].index(max(score_list[0]))
        print(score)
        print(label)

        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        print(epoch, ": ", running_loss)



