"""
problem tanimi: veri sıkıştırması -> autoencoders
veri: FashionMNIST
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
# %% veri yükleme ve ön işleme
transform = transforms.Compose([transforms.ToTensor()]) # görüntüleri tensore çevir ve [0-1] arasına sıkıştırır.

#eğitim ve test veri setinin indir ve yükle
train_dataset=datasets.FashionMNIST(root="./data",train=True,transform=transform,download=True)
test_dataset = datasets.FashionMNIST(root="./data",train=False,transform=transform,download=True)

#batch_size
batch_size = 128

#eğitim ve test veri yükleyicileri oluşturma
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# %% autoencoders geliştirme
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Flatten(), #28x28 ->784 vektör
            nn.Linear(28*28,256), #tam bağlı katman: 784 -> 256
            nn.ReLU(),#aktivasyon fonksiyonu
            nn.Linear(256,64),#tam bağlı katman: 256 -> 64
            nn.ReLU() #aktivasyon fonksiyonu
        )
        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(64,256),#tam bağlı katman: 64 -> 256
            nn.ReLU(), #aktivasyon fonksiyonu
            nn.Linear(256,28*28),#tam bağlı katman: 256 -> 784
            nn.Sigmoid(),#aktivasyon fonksiyonu sigmoid0 0 ile 1 arasıdnda tutmak için 
            nn.Unflatten(1,(1,28,28)) # tek boyutlu çıktıyı tekrardan 28x28 yapar
        )

    def forward(self,x):
        encoded = self.encoder(x) #giriş verisini kodlar
        decoded = self.decoder(encoded)
        return decoded

# %% callback: early stopping

class EarlyStopping: #erken durdurma (callback sınıfı)
    def __init__(self,patience=5,min_delta=0.001):
        self.patience = patience#kaç epoch boyunca gelişme olmazsa durduracağımızı belirleyeceğimiz parametre
        self.min_delta = min_delta#kayıptaki minimum iyileşme miktarı
        self.best_loss = None#en iyi kayıp değerini sakla
        self.counter = 0
    def __call__(self,loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else: #gelişme yoksa sataç artırılır
            self.counter += 1

        if self.counter >= self.patience: #sabit kalan epoch sayısı patience yi aşarsa -> durdur
            return True # traning Durdur
        return False

# %% model traning

#hyperparameters
epochs = 50
learning_rate = 0.003

#model,loss,optimizer tanımlama

model = AutoEncoder()
criterion = nn.MSELoss() #kayıp fonksiyonu
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
early_stopping = EarlyStopping(patience=5,min_delta=0.001) #erken durduma objesi

#eğtim fonksiyonu 
def training(model,train_loader,optimizer,criterion,early_stopping,epochs):
    model.train() #modeli eğitim moduna al
    for epoch in range(epochs):
        total_loss = 0 #epoch başına toplam kayıt
        for inputs,_ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,inputs) #gerçek ile tahmini veriler arasındaki kayıp
            loss.backward() #gradyanları hesapla
            optimizer.step()
            total_loss += loss.item() #her epoch için toplam loss değeri

        avg_loss = total_loss / len(train_loader) #epoch başına ortalaa kayıp hesaplama
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")

        #early stopping
        if early_stopping(avg_loss): #early stopping koşulları sağlanıyorsa 
            print(f"Early Stopping at epoch {epoch+1}")
            break

training(model,train_loader,optimizer,criterion,early_stopping,epochs)

# %% model testing


def compute_ssim(img1,img2,sigma=1.5):
    """
    iki görüntü arasındaki benzerliği hesaplar
    """
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    #görüntü ortalamaları
    mu1 = gaussian_filter(img1,sigma)
    mu2 = gaussian_filter(img2,sigma)

    #
    mu1_sq = mu1**2 # ilk görüntünün ortalamasının karesi
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2

    sigma1_sq = gaussian_filter(img1**2,sigma) - mu1_sq #varyans hesabı
    sigma2_sq = gaussian_filter(img2**2,sigma) - mu2_sq
    sigma12 = gaussian_filter(img1*img2,sigma) - mu1_mu2 #kovaryans hesabı

    #ssim haritası açıklama
    ssim_map = ((2*mu1_mu2 + C1) * (2 * sigma12 + C2/(mu1_sq + mu2_sq + C1)*(sigma1_sq+ sigma2_sq + C2)))
    return ssim_map.mean()
def evuluate(model,test_loader,n_images = 10):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs,_ = batch
            outputs = model(inputs) #model çıktıları
            break
    inputs = inputs.numpy() #numpy a çevir.
    outputs = outputs.numpy() #numpy a çevir.
    fig,axes = plt.subplots(2,n_images,figsize = (n_images,3))
    ssim_scores = [] #ssim skorlarını saklamak için

    for i in range(n_images):
        img1 = np.squeeze(inputs[i]) # orjinal görüntüyü sıkıştır
        img2 = np.squeeze(outputs[i]) # yeniden oluşturulmuş görüntüyü sıkıştır

        ssim_score = compute_ssim(img1,img2) #sssi skoru yani benzerlik hesapla
        ssim_scores.append(ssim_score) # ssim skorunu ekle

        axes[0,i].imshow(img1,cmap="gray")
        axes[0,i].axis("off")

        axes[1,i].imshow(img2,cmap="gray")
        axes[1,i].axis("off")

    axes[0,0].set_title("Orginal")
    axes[1,0].set_title("Decoded image")
    plt.show()

    #average ssim
    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim}")

evuluate(model,test_loader,n_images = 10)
# %%
