import numpy as np 
import pandas as pd
import matplotlib.pylot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
data=pd.read_csv('mnist.csv')

df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)

ann=Perceptron()

ann.fit(x_train,y_train)

pred=ann.predict(x_test)
# Gizli katman boyutları ile aktivasyon lojistik -> 45,90% 92 doğruluk verdi
# Gizli katman boyutları ile aktivasyon relu -> 45,90% 89 doğruluk verdi
# Öğrenme oranı, aktivasyon ve diğer hiper parametrelerin farklı kombinasyonları ile test edin ve doğruluğu ölçün
print(pred)
a=y_test.values
print(a)
count=0
for i in range(len(pred)):
    if pred[i]==a[i]:
        count=count+1
print(count)
len(pred)
print(7224/8400.0)
