import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %% veri setinin içeriye aktarılması
#sınıflandırma problemi: iris veri seti 3 farklı sınıfta ait sınıflandırma problemi
df = pd.read_csv('6_rbfn/dataset/iris.data', header=None)

X = df.iloc[:,:-1].values # ilk 4 sütunu X değişkenine atar
y,_ = pd.factorize(df.iloc[:,-1])

#veriyi standardize et
scaler = StandardScaler()
X = scaler.fit_transform(X)

#train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

def to_tensor(data,target):
    return torch.torch.tensor(data,dtype=torch.float32), torch.tensor(target,dtype = torch.long)
X_train,y_train = to_tensor(X_train,y_train)
X_test,y_test = to_tensor(X_test,y_test)

# %% RBFN modelinin ve rbf_kernel tanımlanması

def rbf_kernel(X,centers,beta):
    return torch.exp(-beta * torch.cdist(X,centers)**2)

class RBFN(nn.Module):
    def __init__(self,num_centers,input_dim,output_dim):
        super(RBFN,self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers,input_dim)) #rbf merkezlerini rastgele baslat
        self.beta = nn.Parameter(torch.ones(1) * 2.0)#beta parametresi rbf in genişliğini kontorl edecek
        self.linear = nn.Linear(num_centers,output_dim)#outputu tam bağlantılı katmana yönlendir.
    def forward(self,X):
        #rbf çekirdek fonksiyonunu hesapla
        phi = rbf_kernel(X,self.centers,self.beta)
        return self.linear(phi)

# %% model traning
num_centers = 10
model = RBFN(input_dim=4,num_centers=num_centers,output_dim=3)

#kayıp fonksiyonu tanımlama ve optimizasyon

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

#modeli eğitelim
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)    
    loss = criterion(outputs,y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


# %% test and evulation

with torch.no_grad():
    y_pred = model(X_test) #test verisi ile tahmin et
    accuracy = (torch.argmax(y_pred,axis=1) == y_test).float().mean().item() #doğruluk hesapla
    print(f"Accuracy: {accuracy}")