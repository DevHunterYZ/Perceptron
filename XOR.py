# Kütüphaneleri yükleyelim.
from random import choice 
from numpy import array, dot, random
# Step fonksiyonu kullanılır.
unit_step = lambda x: 0 if x < 0 else 1
# Eğitim verisi(matris cinsinden)
training_data = [ (array([0,0,1]), 0), (array([0,1,1]), 1), (array([1,0,1]), 1), (array([1,1,1]), 1), ]
# rastgele 3 matris belirler.
w = random.rand(3)
errors = [] 
eta = 0.2 
n = 100
# 100'e kadar döngü oluştur.
for i in range(n): 
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x
for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

from pylab import plot, ylim
ylim([-1,1])
import matplotlib.pyplot as plt
# Hataları çizdirelim.
plt.plot(errors)
# Grafiği çizdirelim.
plt.show()
