"""
sınıflandırma projesi 
positive ve negative commentler oluşan veri seti
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from collections import Counter

#%% verinin tanımlanması ve veri ön işleme
positive_sentences = [
    "This is amazing.",
    "I absolutely love this product.",
    "The quality is outstanding.",
    "Best purchase I have ever made.",
    "I highly recommend this to everyone.",
    "The design is flawless and beautiful.",
    "It works perfectly fine without any issues.",
    "I am completely satisfied with my order.",
    "Exceeded all of my expectations.",
    "Brilliant work, well done.",
    "This made my day much better.",
    "Incredibly user-friendly and intuitive.",
    "Five stars, I would buy this again in a heartbeat.",
    "Truly a masterpiece of engineering.",
    "The customer support team was excellent.",
    "Flawless performance every single time.",
    "I am so happy with the final result.",
    "Perfect in every possible way.",
    "A wonderful addition to my collection.",
    "Works like a absolute charm.",
    "Top-notch quality materials were used.",
    "This is simply the best on the market.",
    "Superb craftsmanship and attention to detail.",
    "I am completely blown away by this.",
    "Exactly what I was looking for all along.",
    "Could not be any happier with this service.",
    "Outstanding value for the price paid.",
    "It looks absolutely stunning in person.",
    "A delightful surprise that I really enjoyed.",
    "Phenomenal service from start to finish.",
    "The details are magnificent and precise.",
    "Highly professional and efficient team.",
    "Such a pleasant experience overall.",
    "I can't stop smiling while using this.",
    "It is a total game changer for my workflow.",
    "Absolutely gorgeous aesthetics.",
    "Very impressive speed and accuracy.",
    "I feel great about this investment.",
    "Simply the best application I have used.",
    "Very smooth and lag-free experience.",
    "The battery life is exceptionally good.",
    "An absolute joy to use every day.",
    "The interface is clean and modern.",
    "I appreciate the fast shipping and secure packaging.",
    "It completely solved my problem.",
    "The graphics are breathtakingly realistic.",
    "A fantastic tool for daily tasks.",
    "Everything is clearly labeled and easy to find.",
    "The sound quality is incredibly crisp.",
    "I am amazed by how lightweight it is.",
    "It provides a very secure feeling.",
    "The tutorial was very helpful and clear.",
    "I love the vibrant colors and contrast.",
    "It fits perfectly and feels very comfortable.",
    "A brilliant solution to a complex issue.",
    "The performance is incredibly stable.",
    "I am very grateful for this update.",
    "It saves me so much time every day.",
    "The setup process was a breeze.",
    "Extremely durable and well-built.",
    "The food was absolutely delicious.",
    "The atmosphere in the restaurant was fantastic.",
    "The movie had a brilliant plot and great acting.",
    "An inspiring and beautifully written book.",
    "The soundtrack is epic and memorable.",
    "The integration with other apps is seamless.",
    "I love the customization options available.",
    "The screen resolution is crystal clear.",
    "It is incredibly soft to the touch.",
    "The camera takes stunning photos.",
    "A highly addictive and fun game.",
    "The instructions were very easy to follow.",
    "It delivers exactly what it promises.",
    "The navigation is logical and straightforward.",
    "I am thoroughly impressed by the efficiency.",
    "The packaging was eco-friendly and neat.",
    "It operates very quietly in the background.",
    "The new features are incredibly useful.",
    "I felt very welcomed and valued as a customer.",
    "The warranty policy is very reassuring.",
    "A perfect gift for any occasion.",
    "The flavor profile is rich and balanced.",
    "The fabric is highly breathable and cool.",
    "It handles multitasking effortlessly.",
    "The connectivity is strong and reliable.",
    "I love the sleek and compact design.",
    "It boots up almost instantly.",
    "The community support is fantastic.",
    "It makes my life so much easier.",
    "The automation features are top tier.",
    "I am delighted with the quick response time.",
    "The controls are very responsive.",
    "It stays cool even under heavy load.",
    "The lighting effects are gorgeous.",
    "A very secure and trustworthy platform.",
    "The documentation is extremely comprehensive.",
    "I enjoyed every single minute of it.",
    "The synchronization is perfectly timed.",
    "It is highly optimized for performance.",
    "A truly spectacular and unforgettable experience.",
    "I am fascinated by how well it works.",
    "This update brings a lot of value.",
    "An exceptional piece of software.",
    "The architecture is very well thought out.",
    "I am consistently impressed by the results.",
    "It significantly boosted my productivity.",
    "The user base is incredibly supportive.",
    "I found it very easy to integrate.",
    "This is exactly what the industry needed.",
    "A prime example of excellent engineering.",
    "The rendering speed is incredibly fast.",
    "It exceeded my wildest expectations.",
    "The modular design is a huge plus.",
    "I love the intuitive drag-and-drop feature.",
    "It keeps my data perfectly organized.",
    "The visual feedback is immediate and satisfying.",
    "A remarkably stable and reliable release.",
    "It provides incredibly deep insights.",
    "The transition was smooth and painless.",
    "I appreciate the continuous updates and fixes.",
    "It has become an essential part of my toolkit.",
    "The minimalist approach is very effective.",
    "It handled the high traffic beautifully.",
    "The error messages are clear and helpful.",
    "I am very fond of the dark mode feature.",
    "It scales perfectly with my growing needs.",
    "The attention to security is commendable.",
    "I love how customizable the dashboard is.",
    "It processes large files in seconds.",
    "The cross-platform compatibility is flawless.",
    "I am amazed by the high-resolution output.",
    "It offers unparalleled flexibility.",
    "The default settings are surprisingly good.",
    "I commend the developers for their hard work.",
    "It instantly elevated the quality of my project.",
    "The haptic feedback feels very natural.",
    "It maintains a consistent frame rate.",
    "I am thrilled with the new capabilities.",
    "It simplifies complex tasks effortlessly.",
    "The cloud sync feature is a lifesaver.",
    "I strongly advocate for this platform.",
    "It requires minimal configuration to start.",
    "The typography is highly legible and elegant.",
    "I am impressed by the low latency.",
    "It blends seamlessly into my daily routine.",
    "The machine learning model is highly accurate.",
    "I find the interface deeply engaging.",
    "It offers a superb balance of power and simplicity.",
    "The memory management is highly optimized.",
    "I appreciate the transparent pricing model.",
    "It gives me complete control over my data.",
    "The onboarding process was wonderfully designed.",
    "It features a robust set of advanced tools.",
    "I am delighted by the responsive layout.",
    "It drastically reduced our operating costs.",
    "The search function is incredibly fast and precise.",
    "I love the seamless third-party integrations.",
    "It operates perfectly in offline mode.",
    "The audio mixing tools are professional grade.",
    "I am very impressed by the backward compatibility.",
    "It delivers a truly immersive experience.",
    "The syntax is clean and easy to read.",
    "I appreciate the detailed analytics provided.",
    "It handles edge cases exceptionally well.",
    "The dynamic range of the display is gorgeous.",
    "I am a huge fan of the intuitive gestures.",
    "It runs flawlessly on older hardware.",
    "The automated backups give me peace of mind.",
    "I love the vibrant and active community forums.",
    "It accurately predicts my next actions.",
    "The color calibration is spot on out of the box.",
    "I am very satisfied with the robust API.",
    "It manages resources very efficiently.",
    "The spatial audio creates a fantastic atmosphere.",
    "I highly praise the accessible design features.",
    "It filters out noise perfectly.",
    "The real-time collaboration is incredibly smooth.",
    "I am grateful for the open-source community behind it.",
    "It renders complex 3D models with ease.",
    "The text-to-speech functionality is highly natural.",
    "I love the dynamic lighting system.",
    "It effortlessly handles concurrent connections.",
    "The biometric security is fast and reliable.",
    "I appreciate the straightforward installation process.",
    "It gracefully recovers from unexpected errors.",
    "The built-in templates are very professional.",
    "I am stunned by the realistic physics engine.",
    "It provides actionable and clear feedback.",
    "The localization is done incredibly well.",
    "I love the clean and logical file structure.",
    "It seamlessly bridges the gap between devices.",
    "The cache mechanism significantly speeds up loading.",
    "I am constantly discovering new, useful features.",
    "It ensures maximum privacy and data protection.",
    "The gesture controls are highly responsive.",
    "I enjoy the gamified learning experience.",
    "It generates highly accurate reports.",
    "The thermal management is exceptionally good.",
    "I am completely satisfied with the battery lifespan.",
    "It offers an intuitive node-based workflow.",
    "The video playback is smooth and artifact-free.",
    "I appreciate the rich text editing capabilities.",
    "It handles extreme conditions without failing.",
    "The layout adapts perfectly to different screen sizes.",
    "I love the sophisticated aesthetic of the UI.",
    "It securely encrypts all incoming and outgoing traffic.",
    "The vector graphics remain sharp at any zoom level.",
    "I am very impressed by the quick turnaround time.",
    "It offers a highly flexible grid system.",
    "The intelligent routing saves so much time.",
    "I admire the ethical approach of the company.",
    "It drastically improved our team's communication.",
    "The automated testing suite is incredibly thorough.",
    "I love how it organizes my media library.",
    "It easily outperforms all its competitors.",
    "The built-in dictionary is extremely helpful.",
    "I am very pleased with the ergonomic design.",
    "It perfectly matches the described specifications.",
    "The drag-and-drop interface is remarkably fluid.",
    "I appreciate the high level of code maintainability.",
    "It successfully bridges legacy and modern systems.",
    "The noise cancellation blocks out everything.",
    "I am a strong supporter of their sustainability efforts.",
    "It beautifully visualizes complex data sets.",
    "The API documentation is a joy to read.",
    "I absolutely recommend this to my colleagues.",
    "It performs complex calculations instantly.",
    "The tactile feedback of the keys is wonderful.",
    "I love the minimalist, clutter-free environment.",
    "It provides a very stable and secure connection.",
    "The voice recognition is stunningly accurate.",
    "I am highly satisfied with the seamless updates.",
    "It consistently delivers high-quality output.",
    "The modular plugins extend its capability perfectly.",
    "I appreciate the proactive customer support.",
    "It has fundamentally changed how I work for the better."
]

negative_sentences = [
    "I do not like this at all.",
    "This is absolutely terrible.",
    "Worst experience I have ever had.",
    "A complete and utter waste of money.",
    "Do not buy this under any circumstances.",
    "I am very disappointed with the outcome.",
    "It broke immediately after one use.",
    "Terrible customer service and rude staff.",
    "The quality is shockingly awful.",
    "I want a full refund right now.",
    "Absolutely horrific and unacceptable.",
    "One star is too generous for this.",
    "Definitely not worth the high price.",
    "It feels very cheap and flimsy.",
    "I hate everything about this product.",
    "Such a massive letdown.",
    "Totally useless for my needs.",
    "The package arrived severely damaged.",
    "It doesn't work as advertised at all.",
    "I deeply regret buying this item.",
    "Very poorly made with cheap materials.",
    "Unacceptable level of quality control.",
    "The core design is fundamentally flawed.",
    "Extremely frustrating to set up and use.",
    "I am never coming back to this store.",
    "This completely ruined my day.",
    "Save your money and look elsewhere.",
    "A complete disaster from start to finish.",
    "Horrible experience with the technical support.",
    "Far below my original expectations.",
    "I am pretty sure this is a scam.",
    "Terrible value for the amount I paid.",
    "Not recommended to anyone.",
    "I am returning this immediately.",
    "It has a very strange and bad smell.",
    "Looks nothing like the picture online.",
    "Very uncomfortable to wear for long periods.",
    "Constant errors and annoying glitches.",
    "The app crashes constantly on my phone.",
    "It arrived completely broken.",
    "The user interface is a confusing mess.",
    "Extremely slow and laggy performance.",
    "The battery drains unbelievably fast.",
    "An absolute nightmare to deal with.",
    "The menu layout is entirely illogical.",
    "Shipping was delayed by several weeks.",
    "It created more problems than it solved.",
    "The graphics look like they are from 2010.",
    "A terrible tool that lacks basic features.",
    "Nothing is labeled, making it hard to use.",
    "The sound is muffled and heavily distorted.",
    "It is way too heavy and bulky.",
    "I feel very unsafe using this platform.",
    "The tutorial was confusing and useless.",
    "The colors are dull and washed out.",
    "It doesn't fit properly and causes pain.",
    "A stupid solution that makes no sense.",
    "The connection drops randomly all the time.",
    "I hate this new forced update.",
    "It wastes so much of my precious time.",
    "The installation was a complete nightmare.",
    "Fragile and breaks at the slightest touch.",
    "The food was cold and completely tasteless.",
    "The restaurant environment was too loud and dirty.",
    "The movie was incredibly boring and predictable.",
    "A poorly written book with a terrible ending.",
    "The soundtrack is extremely annoying.",
    "It fails to sync with any of my other devices.",
    "There are zero customization options.",
    "The screen is pixelated and blurry.",
    "It feels rough and scratches my skin.",
    "The camera takes grainy and dark pictures.",
    "A boring game full of microtransactions.",
    "The manual is written in broken English.",
    "It fails to deliver on every single promise.",
    "Navigating this site is a total headache.",
    "I am shocked by how inefficient this is.",
    "The packaging was excessive and wasteful.",
    "It makes a loud, annoying buzzing noise.",
    "The new features broke the entire system.",
    "I was ignored by the staff for twenty minutes.",
    "Their warranty policy is practically useless.",
    "The worst gift I have ever received.",
    "The flavor is artificial and disgusting.",
    "The material makes me sweat terribly.",
    "It freezes completely when multitasking.",
    "The Bluetooth connection keeps dropping.",
    "I despise the ugly and bulky design.",
    "It takes forever just to boot up.",
    "The community is toxic and unhelpful.",
    "It just makes my workflow much harder.",
    "The automated features never work right.",
    "I have been waiting days for a single reply.",
    "The buttons feel stiff and unresponsive.",
    "It overheats dangerously within minutes.",
    "The lighting is harsh and gives me a headache.",
    "A vulnerable system prone to data leaks.",
    "There is absolutely no documentation provided.",
    "I hated every second of using it.",
    "The audio and video are out of sync.",
    "It uses way too much RAM and CPU.",
    "A truly miserable and frustrating experience.",
    "The server downtime is completely unacceptable.",
    "I am fed up with the constant spam emails.",
    "It deletes my saved files randomly.",
    "The architecture is an unmaintainable mess.",
    "I strongly advise against using this service.",
    "It completely destroyed my previous configuration.",
    "The user base is overwhelmingly hostile.",
    "I found the integration process absolutely agonizing.",
    "This is exactly what the industry should avoid.",
    "A prime example of terrible software development.",
    "The rendering speed is painfully slow.",
    "It failed to meet even my lowest expectations.",
    "The monolithic design makes updates impossible.",
    "I despise the clunky drag-and-drop feature.",
    "It corrupted all of my organized data.",
    "The visual feedback is delayed and confusing.",
    "An extremely unstable and buggy release.",
    "It provides absolutely no useful insights.",
    "The transition was chaotic and stressful.",
    "I hate the lack of meaningful updates and fixes.",
    "It is a completely useless addition to my toolkit.",
    "The interface is overly cluttered and confusing.",
    "It completely crashed under minimal traffic.",
    "The error messages are cryptic and unhelpful.",
    "I strongly dislike the poorly implemented dark mode.",
    "It fails to scale and bottlenecks instantly.",
    "The glaring security flaws are terrifying.",
    "I hate how rigid and restricted the dashboard is.",
    "It struggles to process even the smallest files.",
    "The cross-platform compatibility is a complete joke.",
    "I am disgusted by the low-resolution output.",
    "It offers absolutely zero flexibility.",
    "The default settings are aggressively bad.",
    "I blame the developers for this mess.",
    "It completely ruined the quality of my project.",
    "The haptic feedback feels cheap and rattling.",
    "It constantly drops frames and stutters.",
    "I am furious with the removal of core features.",
    "It overcomplicates simple tasks needlessly.",
    "The cloud sync feature deleted my local copies.",
    "I strongly warn against using this platform.",
    "It requires tedious and endless configuration to start.",
    "The typography is unreadable and poorly spaced.",
    "I am frustrated by the high latency and lag.",
    "It disrupts my daily routine completely.",
    "The machine learning model is biased and inaccurate.",
    "I find the interface incredibly repulsive.",
    "It offers neither power nor simplicity.",
    "The memory leaks make it unusable after an hour.",
    "I despise the deceptive hidden pricing model.",
    "It steals control over my own personal data.",
    "The onboarding process was infuriatingly long.",
    "It lacks any basic essential tools.",
    "I am annoyed by the broken responsive layout.",
    "It drastically increased our operating costs.",
    "The search function is broken and returns garbage.",
    "I hate the complete lack of third-party integrations.",
    "It becomes completely useless in offline mode.",
    "The audio mixing tools are amateurish at best.",
    "I am outraged by the lack of backward compatibility.",
    "It breaks immersion constantly with bugs.",
    "The syntax is convoluted and impossible to read.",
    "I resent the invasive telemetry and tracking.",
    "It crashes spectacularly on edge cases.",
    "The dynamic range of the display is washed out.",
    "I am annoyed by the unresponsive touch gestures.",
    "It fails to run even on top-tier hardware.",
    "The corrupted automated backups ruined my work.",
    "I hate the toxic and elitist community forums.",
    "It constantly misinterprets my commands.",
    "The color calibration is completely ruined out of the box.",
    "I am highly dissatisfied with the broken API.",
    "It hogs system resources unnecessarily.",
    "The spatial audio is disorienting and unnatural.",
    "I heavily criticize the lack of accessibility features.",
    "It amplifies background noise instead of filtering it.",
    "The real-time collaboration constantly desyncs.",
    "I regret contributing to this abandoned open-source project.",
    "It struggles to render basic 3D shapes.",
    "The text-to-speech functionality sounds robotic and creepy.",
    "I hate the glitched dynamic lighting system.",
    "It instantly disconnects upon concurrent connections.",
    "The biometric scanner never recognizes my input.",
    "I despise the convoluted installation process.",
    "It utterly fails to recover from simple errors.",
    "The built-in templates are hideous and outdated.",
    "I am disappointed by the broken physics engine.",
    "It provides vague and useless error feedback.",
    "The localization is clearly machine-translated garbage.",
    "I hate the disorganized and chaotic file structure.",
    "It fails to communicate between devices.",
    "The cache mechanism actually slows down loading times.",
    "I am constantly finding new, game-breaking bugs.",
    "It compromises privacy and leaks user data.",
    "The gesture controls are overly sensitive and erratic.",
    "I hate the forced, annoying gamified learning experience.",
    "It generates completely fabricated reports.",
    "The thermal management is nonexistent and dangerous.",
    "I am appalled by the terrible battery lifespan.",
    "It forces an unintuitive workflow on the user.",
    "The video playback is choppy and full of artifacts.",
    "I despise the limited text editing capabilities.",
    "It catastrophically fails under minimal stress.",
    "The layout breaks entirely on mobile devices.",
    "I hate the ugly, outdated aesthetic of the UI.",
    "It transmits sensitive data in plain text.",
    "The vector graphics appear jagged and pixelated.",
    "I am furious about the agonizingly slow turnaround time.",
    "It forces a rigid, unchangeable grid system.",
    "The broken routing algorithms waste so much time.",
    "I condemn the unethical practices of the company.",
    "It severely hampered our team's communication.",
    "The automated testing suite misses critical bugs.",
    "I hate how it constantly rearranges my media library.",
    "It is easily outperformed by free alternatives.",
    "The built-in dictionary is incredibly limited.",
    "I am in physical pain from the unergonomic design.",
    "It completely misrepresents its actual specifications.",
    "The drag-and-drop interface constantly drops files.",
    "I am disgusted by the spaghetti code maintainability.",
    "It utterly fails to bridge legacy and modern systems.",
    "The noise cancellation creates a painful pressure in my ears.",
    "I strongly condemn their blatant greenwashing efforts.",
    "It fails to visualize even basic data sets correctly.",
    "The API documentation is incomplete and wrong.",
    "I absolutely warn my colleagues to avoid this.",
    "It hangs indefinitely on simple calculations.",
    "The keys feel mushy and unresponsive.",
    "I hate the overly complex, bloated environment.",
    "It drops the connection every five minutes.",
    "The voice recognition fails to understand simple words.",
    "I am incredibly frustrated with the broken updates.",
    "It consistently outputs low-quality garbage.",
    "The modular plugins constantly conflict and crash.",
    "I despise the automated, useless customer support bots.",
    "It has fundamentally ruined my workflow entirely."
]

# veri ön işleme
def preproccess(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text  

# veri seti oluşturma
data = positive_sentences + negative_sentences
labels = [1]*len(positive_sentences)+ [0]*len(negative_sentences)

#veriyi önişleme
data = [preproccess(sentence) for sentence in data]

#vocab oluştur (kelime dağarcı)
all_words = " ".join(data).split()
word_counts = Counter(all_words)
vocab = {word: idx+1 for idx,(word,_) in enumerate(word_counts.items())}
vocab["<PAD>"] = 0 #padding özel token tanımla


#veriyi tensore çevirme
max_len = 15
def sentence_to_tensor(sentence,vocab,max_len=15):
    tokens = sentence.split() #cümlei tokenlea yani kelimelere ayır
    indices = [vocab.get(word,0) for word in tokens] #index alma
    indices = indices[:max_len]
    indices += [0]*(max_len - len(indices))
    return torch.tensor(indices)
X = torch.stack([sentence_to_tensor(sentence,vocab,max_len) for sentence in data])
y = torch.tensor(labels)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% transforme modelinin oluşturulması
class TransformerClass(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_heads,num_layers,hidden_dim,num_classes):
        super(TransformerClass,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1,max_len,embedding_dim))
        self.transformer = nn.Transformer(d_model=embedding_dim, #embedding vektör boyutu
                                          nhead=num_heads,#multihead attention mekanizmasındaki başlık sayısı
                                          num_encoder_layers=num_layers, #transformer encoder katmanı sayısı
                                          dim_feedforward=hidden_dim)#encoder içinde bulunan gizli katman boyutu

        self.fc = nn.Linear(embedding_dim*max_len,hidden_dim)
        self.out = nn.Linear(hidden_dim,num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        embedded = self.embedding(x)+self.positional_encoding
        output = self.transformer(embedded,embedded)
        output = output.view(output.size(0),-1)
        output = torch.relu(self.fc(output))
        output = self.out(output)
        output = self.sigmoid(output)
        return output

#model = TransformerClass(len(vocab),32,4,4,64,1)

# %% traning
vocab_size = len(vocab)
embedding_dim = 32
num_head = 4
num_layes = 4
hiddem_dim = 64
num_classes = 1 #olumlu yorumlar ve olumsuz yorumlar
num_epochs = 30


model = TransformerClass(vocab_size,embedding_dim,num_head,num_layes,hiddem_dim,num_classes)

#loss ve optimizer 
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0005)

#traning
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train.long()).squeeze()
    loss = criterion(output,y_train.float())
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epochs},Loss: {loss}")


#%% test

model.eval()
with torch.no_grad():
    y_pred = model(X_test.long()).squeeze()
    y_pred = (y_pred > 0.5).float()

    y_pred_traning = model(X_train.long()).squeeze()
    y_pred_traning = (y_pred_traning > 0.5).float()


    
    accurary = accuracy_score(y_test,y_pred)
    print(f"Test Accurary: {accurary}")

    accurary_train = accuracy_score(y_train,y_pred_traning)
    print(f"Train Accurary: {accurary_train}")
# %%
