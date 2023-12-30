from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gc

class DisasterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir)]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if "pre" in self.image_files[idx]:
            label = torch.tensor([0])
        else:
            label = torch.tensor([1])

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])
dataset_cgan = DisasterDataset(root_dir='data/PRJ-3563/images', transform=transform)
dataloader_cgan = DataLoader(dataset_cgan, batch_size=8, shuffle=True, pin_memory=True)

for img, _ in dataloader_cgan:
    print(img.shape)
    break


class Generateor(nn.Module):
    def __init__(self):
        super(Generateor, self).__init__()
        # input
        self.linear1 = nn.Linear(100, 1024 * 8 * 8)
        self.bn1 = nn.BatchNorm1d(1024 * 8 * 8)
        self.linear2 = nn.Linear(1, 1024 * 8 * 8)
        self.bn2 = nn.BatchNorm1d(1024 * 8 * 8)  # 8*8
        # 反卷积
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 3, 1, 1)  # 8*8
        self.bn3 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)  # 16*16
        self.bn4 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 32*32
        self.bn5 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 64*64
        self.bn6 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 128*128
        self.bn7 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 256*256
        self.bn8 = nn.BatchNorm2d(32)
        self.deconv7 = nn.ConvTranspose2d(32, 16, 4, 2, 1)  # 512*512
        self.bn9 = nn.BatchNorm2d(16)
        self.deconv8 = nn.ConvTranspose2d(16, 3, 4, 2, 1)  # 1024*1024

    def forward(self, x1, x2):  # x1:noise, x2:y
        x1 = F.relu(self.linear1(x1))
        x1 = self.bn1(x1)
        x2 = F.relu(self.linear2(x2))
        x2 = self.bn2(x2)
        x1 = x1.view(-1, 1024, 8, 8)
        x2 = x2.view(-1, 1024, 8, 8)
        x = torch.cat([x1, x2], axis=1)
        x = F.relu(self.deconv1(x))
        x = self.bn3(x)
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = F.relu(self.deconv3(x))
        x = self.bn5(x)
        x = F.relu(self.deconv4(x))
        x = self.bn6(x)
        x = F.relu(self.deconv5(x))
        x = self.bn7(x)
        x = F.relu(self.deconv6(x))
        x = self.bn8(x)
        x = F.relu(self.deconv7(x))
        x = self.bn9(x)
        x = torch.tanh(self.deconv8(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.linear = nn.Linear(1, 1*1024*1024) #1024*1024
        self.conv1 = nn.Conv2d(4,64,kernel_size=3,stride=2,padding=1) #512*512
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1) #256*256
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1) #128*128
        self.bn2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1) #64*64
        self.bn3 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512,1024,kernel_size=3,stride=2,padding=1) #32*32
        self.bn4 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024*32*32,1024)
        self.fc2 = nn.Linear(1024,1)


    def forward(self, x1, x2):  # x1:y x2:img
        x1 = F.leaky_relu(self.linear(x1))
        x1 = x1.view(-1, 1, 1024, 1024)
        x = torch.cat([x1, x2], dim=1)
        x = F.dropout2d(F.leaky_relu(self.conv1(x)))
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))
        x = self.bn1(x)
        x = F.dropout2d(F.leaky_relu(self.conv3(x)))
        x = self.bn2(x)
        x = F.dropout2d(F.leaky_relu(self.conv4(x)))
        x = self.bn3(x)
        x = F.dropout2d(F.leaky_relu(self.conv5(x)))
        x = self.bn4(x)
        x = x.view(-1, 1024*32*32)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gen = Generateor().to(device)
dis = Discriminator().to(device)
loss = torch.nn.BCELoss()
d_optimizer = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0001)

for step, (img, label) in enumerate(dataloader_cgan):
    label = label.to(torch.float32).to(device)
    img = img.to(torch.float32).to(device)
    break


def generate_and_save_images(model, epoch, noise, label):
    noise = noise.to(torch.float32).to(device)
    label = label.to(torch.float32).to(device)
    predictions = model(noise, label).detach().to(device)
    predictions = predictions.permute(0, 2, 3, 1)
    predictions = (predictions + 1) / 2
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i].cpu().numpy())
        plt.axis('off')
        plt.savefig('img_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


noise_seed = torch.randn(8, 100)
label_seed = torch.randint(0, 2, size=(8, 1))
print(label_seed)

for step, (img, label) in enumerate(dataloader_cgan):
    generate_and_save_images(gen, 1, noise_seed, label_seed)
    break

D_loss = []
G_loss = []


for epoch in range(100):
    D_epoch_loss = 0
    G_epoch_loss = 0
    for step, (img, label) in enumerate(dataloader_cgan):
        label = label.to(torch.float32).to(device)
        img = img.to(torch.float32).to(device)
        size = img.shape[0]
        random_seed = torch.rand(size, 100).to(device)
        # 训练生成器　
        d_optimizer.zero_grad()
        real_output = dis(label, img).to(device)
        d_real_loss = loss(real_output, torch.ones_like(real_output))
        d_real_loss.backward()
        generated_img = gen(random_seed, label).to(device)
        fake_output = dis(label, generated_img.detach()).to(device)
        d_fake_loss = loss(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()
        dis_loss = d_real_loss + d_fake_loss
        d_optimizer.step()
        # 训练生成器
        g_optimizer.zero_grad()
        fake_output = dis(label, generated_img)
        gen_loss = loss(fake_output, torch.ones_like(fake_output))
        gen_loss.backward()
        g_optimizer.step()

        with torch.no_grad():
            D_epoch_loss += dis_loss.item()
            G_epoch_loss += gen_loss.item()

    with torch.no_grad():
        D_epoch_loss /= 174
        G_epoch_loss /= 174
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)
        print("epoch:", epoch)
        print('D_epoch_loss:', D_epoch_loss)
        print('G_epoch_loss:', G_epoch_loss)
        if epoch==99:
            generate_and_save_images(gen, epoch, noise_seed, label_seed)
    gc.collect()
    torch.cuda.empty_cache()

for img, _ in dataset_cgan:
    img = img.permute(1, 2, 0)
    img = (img + 1) / 2
    plt.imshow(img)
