import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(1,300,28*28)
x = np.reshape(x,(28,28))
y = x + 1
plt.plot(x, y)
plt.figure(2)
plt.imshow(x, cmap='gray')
plt.show()
plt.figure(3)
plt.imshow(x, cmap='gray')
# plt.show()
x *= y
plt.imshow(x, cmap='gray')
plt.show()
