"""
lstm ile metin türetme
"""


# %% library
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter # kelime frekanslarını hesaplamak için
from itertools import product #gridsearch için kombinasyon oluşturmak

# %% veri yükleme ve ön işleme

#ürün  yorumları

text= """
Bu ürün beklentimi fazlasıyla karşıladı.
Malzeme kalitesi gerçekten çok iyi.
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.
Fiyatına göre performansı harika.
Kesinlikle tavsiye ederim ve öneririm!
"""
#veri ön işleme: 
#noktalama işaretlerinden kurtul.
#küçük harf dönüşümü.
#kelimeleri böl

words = text.replace(".","").replace("!","").lower().split()

#kelime frekanslarını hesapla ve indexleme oluştur.
word_counts = Counter(words)
vocab = sorted(word_counts,key=word_counts.get,reverse=True) #kelime frekansını büyükten küçüğe doğru sırala
word_to_ix = {word: i for i,word in enumerate(vocab)}
ix_to_word = {i: word for i,word in enumerate(vocab)}

#eğitim verisi hazırlama
data = [(words[i],words[i+1]) for i in range(len(words)-1)]

# %% lstm modeli tanımlama
class LSTM(nn.Module):
    """
    input -> embedding -> lstm -> Linear -> output
    
    
    """
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(LSTM,self).__init__() #bir üst sınıfın constructor'ı çağırma
        self.embedding = nn.Embedding(vocab_size,embedding_dim) #embedding katmanı
        self.lstm = nn.LSTM(embedding_dim,hidden_dim) #LSTM katmanı
        self.fc = nn.Linear(hidden_dim,vocab_size)

    def forward(self,x): #ileri besleme fonksiyonu
        x = self.embedding(x) #input -> embedding
        lstm_out,_ = self.lstm(x.view(len(x),1,-1))
        output = self.fc(lstm_out.view(1,-1)) 
        return output

model = LSTM(len(vocab),embedding_dim=8,hidden_dim=32)

# %% hyperparameter tuning

#kelime listesi -> tensor
def prepare_squence(seq,to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)


#hyperparameter tuning kombnasyonlarını belirle
embedding_size = [8,16] #embedding boyutları
hidden_size = [32,64] #gizli katman boyutları
learning_rate = [0.01,0.005] #öğrenme oranı

best_loss = float("inf") #en düşük kayıp değerini saklamak için değişken
best_param = {} #en iyi parametreleri saklamak içim boş dict

print("Hyperparameter tuning başlıyor...")

#grid search

for emb_size,hidden_size,lr in product(embedding_size,hidden_size,learning_rate):
    print(f"Deneme: Embedding: {emb_size},Hidden: {hidden_size}, Learning Rate: {lr}")

    #modeli tanımla
    model = LSTM(len(vocab),emb_size,hidden_size) #seçilen parametreler ile model oluştur
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = lr) #seçilen lr ile adma optimizerı

    epochs = 50
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0 #epoch başlangıcında kaybı sıfırla
        for word,next_word in data:
            model.zero_grad() #modelin gradyanları sıfırla
            input_tensor = prepare_squence([word],word_to_ix) #girdiği tensore çevir
            target_tensor = prepare_squence([next_word],word_to_ix) #hedef kelimeyi tensore dönüştür
            output = model(input_tensor)
            loss = loss_function(output,target_tensor)
            loss.backward() #geri yayılım işlemi uygula
            optimizer.step() #parametreleri güncelle
            epoch_loss += loss.item()

        if epoch % 10 == 0:
           print(f"Epoch: {epoch}, Loss: {epoch_loss:.5f}")
        total_loss = epoch_loss

    #en iyi modeli kaydet
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {"embedding_size": emb_size,"hidden_dim":hidden_size,"learning_rate":lr}
    print()
print(f"Best params: {best_params}")
            



# %% lstm traning
final_model = LSTM(len(vocab),best_params['embedding_size'],best_params['hidden_dim'])
optimizer = optim.Adam(final_model.parameters(),lr=best_params['learning_rate'])

print("Final Model training")

epochs = 100
for epoch in range(epochs):
    epoch_loss = 0
    for word,next_word in data:
        final_model.zero_grad()
        input_tensor = prepare_squence([word],word_to_ix)
        target_tensor = prepare_squence([next_word],word_to_ix)
        output=final_model(input_tensor)
        loss = loss_function(output,target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Final Model Epoch: {epoch}, Loss: {epoch_loss:.5f}")

# %% test ve değerlendirme

#kelime tahmini fonksiyonu: baslangıc kelimesi ve n adet kelime uretimini sağla
def predict_sequence(start_word,num_words):
    current_word = start_word #şuanki kelime başlangıç kelimsi olarak ayarlanır
    output_sequence = [current_word] #çıktı dizisi

    for _ in range(num_words): #belirtilen sayıda kelime tahmini
        with torch.no_grad(): #gradyan hesabı yapmadan
            input_tensor=prepare_squence([current_word],word_to_ix)#kelimeden tensor dönüşümü
            output=final_model(input_tensor)
            predicted_idx = torch.argmax(output).item() #en yüksek olasılığa sahip kelimenin indexi
            predicted_word = ix_to_word[predicted_idx] #indexe karşılık gelen kelimeyi return eder
            output_sequence.append(predicted_word)
            current_word = predicted_word #bir sonraki tahmin için mevcut kelimeleri güncelle
    return output_sequence #tahmim edilen kelime dizisi return edilir
start_word = "ürün"
num_predictions = 1
predicted_sequence = predict_sequence(start_word,num_predictions)
print(" ".join(predicted_sequence))
# %%
