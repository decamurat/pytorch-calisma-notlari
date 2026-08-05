"""
image generation: MNIST veri seti
"""
from tkinter import W

from sympy import true
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# %% veri seti hazırlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128 #mini batch size
image_size = 28
image_dim = image_size*image_size

transform = transforms.Compose([
    transforms.ToTensor(), # görüntüyü tensore çevir
    transforms.Normalize((0.5,),(0.5,)) # normalizsayon -> -1 ile 1 arasında sıkıştır
])

#mnist veri seti yükle
dataset = datasets.MNIST(root="./data",train=True,transform=transform,download=True)

#verisetinin batchler halinde yüklenmesi
dataLoader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

# %% Discriminator oluştur
class Discriminator(nn.Module): #ayırt edici: generatorun üretmiş olduğu görüntüleri gercek mi sahte mi
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim,1024), #input: image_size, 1024: nöron sayısı yani bu layerın outputu
            nn.LeakyReLU(0.2), #aktivasyon fonksiyonu ve 0.2'lik eğim
            nn.Linear(1024,512), #1024'ten 512 düğüme
            nn.LeakyReLU(0.2),
            nn.Linear(512,256), #512'den 256 düğüme
            nn.LeakyReLU(0.2),
            nn.Linear(256,1), #256'dan tek bir çıktı gercek mi sahte mi
            nn.Sigmoid() #çıktıyı 0-1 arasına getir.
        )
    def forward(self,img):
       return self.model(img.view(-1,image_dim)) #görüntüyü düzleştirerek modele ver
# %% Generator oluştur
class Generator(nn.Module): #görüntü üreten
    def __init__(self,z_dim):
        super(Generator,self).__init__() 
        self.model = nn.Sequential(
            nn.Linear(z_dim,256),#girişten 256 düğüme tam bağımlı katman
            nn.ReLU(),
            nn.Linear(256,512), #256'dan 512 düğüme
            nn.ReLU(),
            nn.Linear(512,1024),#512'den 1024 düğüme
            nn.ReLU(),
            nn.Linear(1024,image_dim), #1024'den 784'e(28*28) çevirim
            nn.Tanh() #çıkış aktivasyon fonksiyonu
        )
    def forward(self,x):
        return self.model(x).reshape(-1,1,28,28) #çıktıyı 28x28 boyutuna çevirir.

# %% GAN traning
#hyperparameters
learning_rate=0.0002
z_dim=100#rastgele gürültü vektör boyutu(noise görüntüsü)
epochs=50

#model baslatma: generator ve discriminator tanımmla
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

#kayıp fonksiyonu ve optimizasyon algoritmalarının tanımlanması

criterion = nn.BCELoss()#binary cross entropy
g_optimizer = optim.Adam(generator.parameters(),lr=learning_rate,betas=(0.5,0.999)) #betas learning rate'in değişme hızını belirler
d_optimizer = optim.Adam(discriminator.parameters(),lr=learning_rate,betas=(0.5,0.999))

#eğitim döngüsü başlatılması
for epoch in range(epochs):

    #görüntülerin yüklenmesi
    for i,(real_imgs,_) in enumerate(dataLoader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0) # [128,28,28] -> 128 bath size ,mevcut batchin boyutunu al (tüm batchler 128 olmaz son batch az olabilir o yüzde buradan tanımlanmalı.)
        real_labels = torch.ones(batch_size,1).to(device) #gerçek görüntüleri 1 olarak etiketle
        fake_labels = torch.zeros(batch_size,1).to(device) #sahte görüntüleri 0 olarak etiketle

        #discriminator eğitimi
        z = torch.randn(batch_size,z_dim).to(device) #rastegele gürültü üret
        fake_imgs = generator(z) #generator ile sahte görüntü oluştur
        real_loss = criterion(discriminator(real_imgs),real_labels) #gerçek görüntü kayıbı
        fake_loss = criterion(discriminator(fake_imgs.detach()),fake_labels) #sahte görüntü kayıbı
        d_loss = real_loss + fake_loss #discriminator kaybı toplamı

        d_optimizer.zero_grad() # gradyanları sıfırla
        d_loss.backward() # geriye yayılım
        d_optimizer.step() #parametreleri güncelle
        
        #generator eğitimi
        g_loss = criterion(discriminator(fake_imgs), real_labels) #generator kaybı
        g_optimizer.zero_grad() #gradyanları sıfırla
        g_loss.backward()#geri yayılım
        g_optimizer.step() # parametleri güncelle

    print(f"Epoch: {epoch+1}/{epochs}, d_loss: {d_loss.item():.3f}, g_Loss: {g_loss.item():.3f}")


# %% model testing and performance evulations

#rastele gürültü ile görüntü oluşturma

with torch.no_grad():
    z = torch.randn(16,z_dim).to(device) #16 adet rastgele gürültü oluştur
    sample_imgs = generator(z).cpu() #generator ile sahte goruntu oluşturma
    grid = np.transpose(utils.make_grid(sample_imgs,nrow=4,normalize=True), (1,2,0))
    plt.imshow(grid)
    plt.show()







#type:ignore
# %%
