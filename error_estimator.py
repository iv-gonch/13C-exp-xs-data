import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

x = np.arange(0, 10, 1e-3)   
y = np.arange(0, 500, 0.5)
X, Y = np.meshgrid(x, y)

def create_gaussian_2d(x0, y0, sigma_x, sigma_y, r):
    """
    Создает двумерное распределение Гаусса
    """
    # Создаем ковариационную матрицу
    cov = [[sigma_x**2, r * sigma_x * sigma_y],
           [r * sigma_x * sigma_y, sigma_y**2]]
    
    # Создаем объект многомерного нормального распределения
    rv = multivariate_normal([x0, y0], cov)

    pos = np.dstack((X, Y))
    
    # Вычисляем значения функции распределения
    Z = rv.pdf(pos)

    return Z


# Параметры распределения
df = pd.read_csv('error_data/6.csv')

en_col = 'EN MEV'
# err_en_col = 'ERR-EN (EV) 1.1'
err_en_col = 'EN-RSL-FW MEV'

data_col = 'DATA MB'
# err_data_col = 'ERR-T (B) 0.911'
err_data_col = 'ERR-S PER-CENT'

array_x0 = df[en_col].values
array_y0 = df[data_col].values

sigma_x = df[err_en_col].values*array_x0    # стандартное отклонение по x
sigma_y = df[err_data_col]  # стандартное отклонение по y
r = np.zeros_like(array_x0) # коэффициент корреляции

Z_sum = np.zeros_like(create_gaussian_2d(x0=0, y0=0, sigma_x=1, sigma_y=1, r=0))

for i in range(len(array_x0)):
    # Создаем распределение
    Z = create_gaussian_2d(array_x0[i], array_y0[i], sigma_x[i], sigma_y[i], r[i])
    Z_sum += Z

# Создаем график
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. График с imshow
im1 = ax1.imshow(Z_sum, origin='lower', 
                extent=[0, 10, 0, 500],
                cmap='viridis', aspect='auto')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
# ax1.set_title(f'2D Gaussian Distribution (imshow)\n'
            #  f'μ=({x0},{y0}), σ=({sigma_x},{sigma_y}), r={r}')
ax1.grid(True, alpha=0.3)
plt.colorbar(im1, ax=ax1, label='Probability Density')

# 2. График с контурами поверх imshow
im2 = ax2.imshow(Z_sum, origin='lower',
                extent=[0, 10, 0, 500],
                cmap='viridis', aspect='auto')
# Добавляем контурные линии
contour = ax2.contour(X, Y, Z_sum/np.max(Z_sum), levels=[0, 0.0001, 0.33, 0.5, 0.685, 0.954, 0.996, 0.9999], colors='white', alpha=0.7)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
# ax2.set_title(f'With Contour Lines\n'
            #  f'μ=({x0},{y0}), σ=({sigma_x},{sigma_y}), r={r}')
ax2.grid(True, alpha=0.3)
plt.colorbar(im2, ax=ax2, label='Probability Density')

plt.tight_layout()
plt.show()

# Выводим параметры
print(f"Параметры распределения:")
# print(f"Центр: ({x0}, {y0})")
print(f"Стандартные отклонения: sigma_x = {sigma_x}, sigma_y = {sigma_y}")
print(f"Коэффициент корреляции: r = {r}")