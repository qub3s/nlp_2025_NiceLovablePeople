# run pip install matplotlib seaborn

import matplotlib
import seaborn

class_distribution = [376, 125, 134, 369, 479, 1757, 305, 140, 4, 10, 540, 33, 118, 14, 50, 28, 313, 522, 49, 202, 2082, 552, 240, 2705, 425, 63]
print(class_distribution)
plt.axvline(x=sum(class_distribution)/26, color='red', linewidth=1, label='Threshold')
plt.show()