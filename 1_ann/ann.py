'''
MNIST veri seti ile rakam sınıflandırma prjesi
ANN: Yapay Sinir Ağları
'''

#Library
import torch # Tensor İşlemleri
import torch.nn as nn #Yapay Sinir Ağı Katmanlarını tanılamak için
import torch.optim as optim #Optimizasyon algoritmalarını içeren modul
import torchvision #Görüntü işleme ve predefined modelleri içerir.
import torchvision.transforms as transforms #Görüntü Dönüşümleri yapmak
import matplotlib.pyplot as plt #Görselleştirme

#Optional: Cihazı Belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DataLoading
def get_data_loaders(batchsize = 64): #Her iterasyonda işlenecek veri miktarı(batchsize)
    transform=transforms.Compose([
        transforms.ToTensor(),#Görüntüyü tensore çevirir ve 0-255 -> 0-1 arası ölçeklendir
        transforms.Normalize((0.5,),(0.5,)) #piksel değerlerini -1 ile 1 arasına ölçekler
    ])

    #MNIST verisetini indir ve eğitim test kümelerini oluştur
    train_set = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform) #./ bu klasörün içi demekmiş
    test_set = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform) #./ bu klasörün içi demekmiş

    #PyTorch veri yükleyicisini oluştur
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batchsize,shuffle=True)
    test_loader = torch.utils.data.DataLoader(train_set,batch_size=batchsize,shuffle=False)
    return train_loader,test_loader
train_loader,test_loader = get_data_loaders()

#Data Visualization
def visualize_samples(loader,n):
    images,labels = next(iter(loader)) #İlk batchten görüntü ve etiketleri alalım
    fig, axes = plt.subplots(1, n, figsize=(10, 5)) #n farklı görüntü için görselleştirme alanı
    for i in range(n):
        axes[i].imshow(images[i].squeeze(),cmap="gray") #Görseli gri göster #type: ignore
        axes[i].set_title(f"Label: {labels[i].item()}")#Sınıf etiketini başlık olarak yaz #type: ignore
        axes[i].axis("off") #Eksenleri Gizle #type: ignore
    plt.show()
visualize_samples(train_loader,4)

#Define ann model
class NeuralNetwork(nn.Module): #pytorchun nn.module sınıfından miras alıyor.
    def __init__(self): #nn inşa etmek için gerekli olan bileşenleri tanımla
        super(NeuralNetwork,self).__init__()

        self.flatten = nn.Flatten() #Elimizde bulunan görüntüleri(2D)#elimizde bulunan görüntüleri vektör haline 1D haline

        self.fc1 = nn.Linear(28*28,128)    #ilk tam bağlı katmanı oluştur: 784 = input_size , 128 = output_size

        self.relu = nn.ReLU()#aktivasyon fonksiyonu oluştur

        self.fc2 = nn.Linear(128,64)#ikinci tam bağlı katmanı oluştur: 128=input_size , 64 = output_size
        
        self.fc3 = nn.Linear(64,10)#çıktı katmanı : 64=input_size , 10=output_size(0-9 etiketleri var 0-9 arasındanki sayıları anlmaya çalıştımız için birini seçmek zorunda 0-9 arası sayılar)

    def forward(self,x): #forward propagarion: ileri yayılım ,giriş olarak x = görüntü alsin

        #initial x = 28*28 lik bir görüntü -> düzleştir 784 vektör haline getir
        x = self.flatten(x)
        x = self.fc1(x) #birinci bağlı katman
        x = self.relu(x) #aktivasyon fonksiyonu
        x = self.fc2(x) #ikinci bağlı katman
        x = self.relu(x) #aktivasyon fonksiyonu
        x = self.fc3(x) #output katmanı
        return x
#Create Model & compile
model = NeuralNetwork().to(device)

#Kayıp fonksiyonu ve optimizasyon algoritmasını belirle
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), #multi class clasification problems loss function
    optim.Adam(model.parameters(),lr = 0.001) #update weights with adam
)
criterion,optimizer = define_loss_and_optimizer(model)
#Train
def train_model(model,train_loader,criterion,optimizer,epochs=10):
    model.train() #modelimii eğitim moduna alalım

    train_losses = []#her bir epoch sonucunda elde edilen kayıpları(loss) saklamak için liste tanımla

    for epoch in range(epochs): #belirtilen epoch sayısı kadar eğitim yapalım
        total_loss = 0 #Toplam loss değeri

        for images,labels in train_loader:#tüm eğitim verileri üzerinde iterasyon gerçekleştir
            images,labels = images.to(device),labels.to(device) #verileri cihaza taşı

            optimizer.zero_grad()#gradyanları sıfırla
            predictions = model(images)#modeli uygula,forward propagation

            loss = criterion(predictions,labels)#loss hesaplama -> y_prediction ile y_real arasında

            loss.backward() #geri yayılım yani gradyan hesaplama

            optimizer.step()#update weights (Ağırlkları güncelle)

            total_loss = total_loss + loss.item()
        avg_loss = total_loss / len(train_loader) #Ortalama kayip hesaplama
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs},Loss: {avg_loss:.3f} ")
    #loss graph
    plt.figure()
    plt.plot(range(1,epochs + 1),train_losses,marker = "o",linestyle= "-",label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
train_model(model,train_loader,criterion,optimizer,epochs=15)

#Test   
def test_model(model,test_loader):
    model.eval() #Modelimizi değerlendirme moduna al
    correct = 0 #Doğru tahmin sayacı
    total = 0 #toplam veri sayacı

    with torch.no_grad(): #gradyan hesaplama gereksiz olduğundan kapattık.
        for images , labels in test_loader: #test veri kümesini döngüye al
            images, labels = images.to(device), labels.to(device)
            predicions = model(images)
            _,predicted=torch.max(predicions,1) # en yüksel olasıklı sınfın etiketini bul.
            total += labels.size(0) #toplam veri sayısını güncelle
            correct += (predicted == labels).sum().item() #doğru tahminleri say
    print(f"Test Accuracy,{100*correct/total:.3f}%")
test_model(model,test_loader)

if __name__ == "__main__":
    train_loader,test_loader = get_data_loaders()
    visualize_samples(train_loader,5)
    model = NeuralNetwork().to(device)
    criterion,optimizer = define_loss_and_optimizer(model)
    train_model(model,train_loader,criterion,optimizer)
    test_model(model,test_loader)

