# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import glob
import os
import math
import matplotlib.pyplot as plt

def interpolate_dataframe(df, Prusachenko_flag):
    """
    Интерполирует датафрейм и создает эквидистантные точки по EN
    
    Параметры:
    df: pandas.DataFrame с колонками 'DATA (B) 0.1' и 'EN-CM (EV) 1.1'
    num_points: количество точек для эквидистантного распределения
    
    Возвращает:
    pandas.DataFrame с интерполированными данными
    """
    # Извлекаем данные
    # Названия колонок из файла A0613003.csv
    data_col = 'DATA (B) 0.1'
    en_col = 'EN (EV) 1.1'
    
    # Удаляем строки с NaN значениями
    df_clean = df.dropna(subset=[data_col, en_col])
    
    # Получаем значения
    x = df_clean[en_col].values
    y = df_clean[data_col].values

    # Создаем функцию интерполяции (кусочно-линейная)
    f = interp1d(x, y, kind='linear', fill_value)
    
    # Создаем эквидистантные точки по EN
    x_new = np.arange(math.floor(x.min()/1e4)*1e4, math.ceil(x.max()/1e4)*1e4, 1e4)
    
    if Prusachenko_flag: 
        arr1 = np.arange(208e4, 240e4, 1e4)
        arr2 = np.arange(296e4, 339e4, 1e4)
        arr3 = np.arange(465e4, 612e4, 1e4) 
        x_new = np.concatenate([arr1, arr2, arr3])

    # Интерполируем значения
    y_new = f(x_new)
    
    # Создаем новый датафрейм
    result_df = pd.DataFrame({
        'EN (MeV)': x_new*1e-6,
        'XS (mb)': y_new*1e3
    })
    
    return result_df

def process_all_csv_files(directory_path):
    """
    Обрабатывает все CSV файлы в директории
    
    Параметры:
    directory_path: путь к директории с CSV файлами
    num_points: количество точек для эквидистантного распределения
    
    Возвращает:
    список интерполированных датафреймов
    """
    # Находим все CSV файлы в директории
    csv_files = [
        '0.csv', # H.W.Drotleff et.al
        '1.csv', # J.K.Bair et.al
        '2.csv', # S.E.Kellogg et.al

        '3.csv', # R.B.Walton et.al
        '4.csv', # K.K.Sekharan et.al
        '5.csv', # C.N.Davids et.al

        # # 'F0786004.csv', # S.Harissopulos et.al 
        '6.csv'  # P.S.Prusachenko et.al
    ]
    
    dataframes = []
    
    for csv_file in csv_files:

        try:
            # Читаем CSV файл
            df = pd.read_csv(csv_file)
            
            # Интерполируем
            if "6" in csv_file:
                Prusachenko_flag = True
            else:
                Prusachenko_flag = False

            interpolated_df = interpolate_dataframe(df, Prusachenko_flag)
            
            # Добавляем в список
            dataframes.append(interpolated_df)
            
            print(f"Обработан файл: {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"Ошибка при обработке файла {csv_file}: {e}")
    
    return dataframes

def main():
    # Путь к директории с данными
    data_directory = "."
    
    # Обрабатываем все CSV файлы
    dataframes = process_all_csv_files(data_directory)
    # Проверяем, что датафреймы были созданы
    if len(dataframes) > 0:
        print(f"\nСоздано {len(dataframes)} интерполированных датафреймов")
        
        for i in range(7):
            # Интерполируем первый датафрейм (dataframes[0]) как запрошено пользователем
            df_interpolated = dataframes[i]
            
            # Сохраняем результат в новый CSV файл
            output_file = "interpolated_" + str(i) + ".csv"
            df_interpolated.to_csv(output_file, index=False)
            print(f"Первый интерполированный датафрейм сохранен в {output_file}")
            
            # Выводим первые несколько строк для проверки
            # print("\nПервые 10 строк интерполированного датафрейма[0]:")
            # print(df_0_interpolated.head(10))
            
            # Выводим последние несколько строк для проверки
            # print("\nПоследние 10 строк интерполированного датафрейма[0]:")
            # print(df_0_interpolated.tail(10))

    else:
        print("Не удалось создать ни одного датафрейма")

if __name__ == "__main__":
    main()