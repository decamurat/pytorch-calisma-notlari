"""
transfer learning: çiçeklerin sınıflandırması 102 farklı etiketten oluşan sınıflandırma
mobilnet ile transfer learning
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm #eğitim sürecini izlemek için kullandığımız bir ilerleme çubuğu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
# %% veri yükleme ve data augmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#veri dönüşümü
#1- klasik dönüşümler: normalizasyon,tensor dönüşümleri
#2- mobilenete uygun giriş boyutu ayarlanması

transform_train = transforms.Compose([
    transforms.Resize((224,224)), #mobilenet input size
    transforms.RandomHorizontalFlip(),#görüntüleri yatay çevirip veri artırımı
    transforms.RandomRotation(10), # görüntleri rastgele 10 dereceye kadar çevirir.
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1), # renk varyasyonları
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5,)) #piksel değerlerini normalize
])  

transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5,)) 
])

#oxford flowers 102 veri seti yükleme

train_dataset = dataset.Flowers102(root="./data",split="train",transform=transform_train,download=True)
test_dataset = dataset.Flowers102(root="./data",split="val",transform=transforms_test,download=False)


#rastgele 5 örnek seçimi
indices = torch.randint(len(train_dataset),(5,))
samples = [train_dataset[i] for i in indices] #type: ignore

#görselleştirme
fig,axes = plt.subplots(1,5,figsize=(15,5))
for i,(image,label) in enumerate(samples):
    image = image.numpy().transpose((1,2,0)) #tensor goruntu formatına çevir.
    image = (image*0.5 + 0.5)
    axes[i].imshow(image)
    axes[i].set_title(f"Label: {label}")
    axes[i].axis("off")
plt.show()

#data loader

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
# %% transfer learning tanımlama ve fine tunning ve model kaydetme

# mobilenet v2 yükleme
model = models.mobilenet_v2(pretrained= True) #pretrained = true -> önceden eğitilmiş ağırlıkları kullan

#sınıflandırıcı katmanı ekleme
num_ftrs = model.classifier[1].in_features #mevcut sınıflandırıcının giriş özellikleri
model.classifier[1] = nn.Linear(num_ftrs,102) # son katmanı oxford flower 102 için değiştir.
model = model.to(device)


# kayıp fonksiyonu ve optimizer tanımlama

criterion = nn.CrossEntropyLoss() #çok sınıflı sınıflandırma için çapraz entropi kullanımı
optimizer = optim.Adam(model.classifier[1].parameters(),lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)

# model traning

epochs = 3
for epoch in tqdm(range(epochs)):
    model.train() #model eğitim modu
    running_loss = 0.0 # toplam kayıp değeri
    for images,labels in tqdm(train_loader):
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad() #gradyan sıfırla
        outputs = model(images) #model ile tahmin
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch: {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

#modeli kaydet
torch.save(model.state_dict(),"mobilenet_flowers102.pth")

# %% test ve değerlendirme 

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images,labels in tqdm(test_loader):
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


#karmaşıklık matrisi
cm = confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(12,12))
sns.heatmap(cm,annot=True,cmap="Blues")
plt.xlabel("Predited")
plt.ylabel("Real")
plt.title("Karmaşıklık matrisi")
plt.show()

print(classification_report(all_labels,all_preds))
# %%
