'''
Problem Tanımı: CIFAR10 veriseti Sınıflandırma
CNN
'''

# %% import libraries
import torch
import torch.nn as nn #sinir ağı katmanları için
import torch.optim as optim #optimizasyon
import torchvision #görüntü işleme
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load dataset
def get_data_loader(batch_size=64): #her iterastonda işlenecek veri sayısı
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #rgb kanallarını normalize et.
    ])
    #CIFAR10 veriseti indir ve eğitim test veri setini oluştuur.
    train_set = torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

    #dataloader
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)

    return train_loader,test_loader

# %% visualize dataset
def imshow(img):
    #verileri normalize etmeden önce geri dönüştür.
    img = img/2 + 0.5 #normalize işleminin tersi
    np_img = img.numpy() #tensor dan numpy array e dön
    plt.imshow(np.transpose(np_img,(1,2,0))) #3 kanal için renkleri doğru şekilde gösterme
    plt.show()

def get_sample_images(train_loader): #veri kümesinden örnek görselleri almak için fonksiyon
    data_iter = iter(train_loader)
    images,labels = next(data_iter)
    return images,labels

def visualize(n):
    train_loader,test_loader = get_data_loader()

    # 3 tane veri görselleştirme
    images,labels = get_sample_images(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1,n,i+1)
        imshow(images[i]) #görseli görselleştir
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()
# %%build CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1) #in_channels=rgb old için 3 ,out_channels=filtre sayısı, kernel_Size 3x3,padding matris kenarlarına 0 ekler bilgi kaybını azaltır.
        self.relu = nn.ReLU() #aktivasyon fonksiyonu
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) # 2x2 boyutunda pooling katmanı,stride filtre işlemlerinde ne kadar kayacağını belirler
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1) #64 filtereli ikinci convolution layer
        self.dropout = nn.Dropout(0.2) #dropout %20 oranında çalışır
        self.fc1 = nn.Linear(64*8*8,128) # 64 filtreli 8x8 çıktısı olan giris:4096 , output: 128
        self.fc2 = nn.Linear(128,10) #output layer 10 farklı etiket olduğu için 10

        #image (3,32x32) -> conv (padding 1 olduğu için boyut değişmez o yüzden 32x32) -> ReLU (32x32) -> pool (16x16)
        #conv(16x16) -> ReLU (16x16) -> pool (8x8) -> image (8x8)



    def forward(self,x):
        """
        image (3,32x32) -> conv (padding 1 olduğu için boyut değişmez o yüzden 32x32) -> ReLU (32x32) -> pool (16x16)
        conv(16x16) -> ReLU (16x16) -> pool (8x8) -> image (8x8)
        flatten
        fc1 -> relu -> dropout
        fc2 -> output
        """

        x = self.pool(self.relu(self.conv1(x))) # ilk convulation layer
        x = self.pool(self.relu(self.conv2(x))) # ikinci convulation layer
        x = x.view(-1,64*8*8) # flatten
        x = self.dropout(self.relu(self.fc1(x)))  # fully connected layer   
        x = self.fc2(x) #output 
        return x   
model = CNN().to(device) 

# define loss function and optimizer
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), #multi classification problem
    optim.SGD(model.parameters(),lr = 0.001,momentum=0.9) #momentum local minimum engellemesi engellenir ve hızlandırır
)

# %% traning
def train_model(model,train_loader,criterion,optimizer,epochs=5):

    model.train() #modeli eğitim moduna alalım.
    train_losses = [] #loss değerlerini saklamak için liste oluştur.
    
    for epoch in range(epochs): #for döngüsü oluştur belirtilen epoch sayısı kadar
        total_loss = 0 #toplam loss değerini saklamak için total_loss tanımla
        for images,labels in train_loader: #for döngüsü tüm eğitim veri setine taramak için
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad() #gradyanları sıfırla
            output = model(images) #forward pro. (predictions) , output=etiketleri
            loss = criterion(output,labels) #loss değeri hesapla
            loss.backward() #geri yayılım(gradyan hesaplama)
            optimizer.step() #öğrenme = parametre yani ağırlık güncelleme

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

    #Kayıp (Loss) Grafiği
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker="o",linestyle="-",label= "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Traning Loss")
    plt.legend()
    plt.show()

# %% test
def test_model(model,test_loader,dataset_type):
    model.eval() #değerlendirme modu
    correct = 0 #doğru tahmin sayacı
    total = 0 #toplam veri sayacı

    with torch.no_grad(): #grad hesaplamasını kapat
        for images,labels in test_loader: #test veri setini kullanarak değerlendirme
            images,labels = images.to(device),labels.to(device)

            outputs=model(images) #predictions
            _,predicted = torch.max(outputs,1)#en yüksek olasılıklı sınıfı seç
            total += labels.size(0) #toplam veri sayısı
            correct += (predicted == labels).sum().item() #doğru tahminleri say

        print(f"{dataset_type} Accuary: {100 * correct / total}%") #doğruluk oranını ekrana yazdır

#main
if __name__ == "__main__":
    #veri seti yükleme
    train_loader,test_loader=get_data_loader()

    #training
    model = CNN().to(device)

    criterion,optimizer=define_loss_and_optimizer(model)
    train_model(model,train_loader,criterion,optimizer,epochs=1)

    #test
    test_model(model,test_loader,dataset_type = "Test")
    test_model(model,train_loader,dataset_type = "Train")




