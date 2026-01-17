# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def plot_all_dataframes(directory_path):
    """
    Строит график для всех интерполированных датафреймов на одном графике
    
    Параметры:
    directory_path: путь к директории с интерполированными CSV файлами
    """
    # Находим все корректные CSV файлы в директории
    csv_files = [
        '1000_JENDL.csv',
        '100_Mohr.csv', 
        '2_Bair_1973.csv', 
        '7_Sekharan_1967.csv',
        '6_Brandenburg_2023.csv', 

        '5_Walton_1957.csv', 
        '10_Prusachenko_2022.csv',
        '4_Febbraro_2020.csv', 

        '1_Drotleff_1993.csv', 
        '3_Kellogg_1989.csv', 
        '8_Davids_1968.csv',
        # '8_Harissopulos_2005_needs_correction.csv',
        '11_Gao_2022.csv'
    ]

    # Имена исходных файлов для легенды
    original_files = [# csv_files
        "JENDL", 
        "S.Harissopulos & P.Mohr",
        "J.K.Bair et.al.", 
        "K.K.Sekharan et.al.", 
        "K. Brandenburg et.al.", # надо перевести в ЛСО из СЦМ

        "R.B.Walton et.al.", 
        r"$(\alpha, n_0)$ P.S.Prusachenko et.al.", 
        r"$(\alpha, n_0)$ M. Febbraro et.al.",
        
        "H.W.Drotleff et.al.", # надо перевести в ЛСО из СЦМ
        "S.E.Kellogg et.al.", 
        "C.N.Davids et.al.", 
        # "S.Harissopulos et.al.", # нет поправки Питера Мора, исключаю из рассмотрения
        "B.Gao et.al."
    ]
    
    # Сортируем файлы по номеру
    # csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if len(csv_files) == 0:
        print("Не найдено корректных CSV файлов")
        return

    # Создаем логарифмический график для сравнения
    plt.figure(figsize=(16, 8))
    # Цвета для разных линий
    colors = plt.cm.tab20(np.linspace(0, 1, 13))
    # Обрабатываем каждый файл 
    for i, csv_file in enumerate(csv_files): 
        try: 
            # Читаем CSV файл 
            df = pd.read_csv(directory_path + csv_file.split(".")[0] + "_corrected.csv") 
            print(directory_path + csv_file.split(".")[0] + "_corrected.csv")
            # Получаем имя для легенды
            if i < len(original_files):
                label = original_files[i]
            else:
                label = f"Dataset {i}"
            # Строим график
            if label == "JENDL":
                plt.plot(df['Ea (eV)'], df['XS (b)'] * 100**i, "-", 
                        color=colors[i], linewidth=5, label=label, alpha=0.5)
            elif label == "K. Brandenburg et.al.":
                plt.plot(df['Ea (eV)'], df['XS (b)'] * 100**i, ".-", 
                        color=colors[i], linewidth=2, label= "K. Brandenburg et.al." + r" $\times 10^{" + str(2*i) + r"}$", alpha=0.8)
            else:
                plt.plot(df['Ea (eV)'], df['XS (b)'] * 100**i, ".-", 
                        color=colors[i], linewidth=2, label=label + r"$ \times 10^{" + str(2*i) + r"}$", alpha=0.8)
            
            print(f"На график добавлен файл: {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"Ошибка при обработке файла {csv_file}: {e}")
    
    # Настройки графика
    plt.xlabel('Alpha Energy (eV)', fontsize=14)
    plt.ylabel('Cross Section (b)', fontsize=14)
    plt.title('Cross Sections for All Datasets', fontsize=16)
    plt.legend(fontsize=12)
        
    # Логарифмическая шкала по Y для лучшего отображения
    plt.yscale('log')
    plt.xlim(0, 8e6)
    plt.tick_params(axis='both')
    plt.grid(visible=True, alpha=0.3, which='both', axis='both')
    # Улучшаем оформление
    plt.tight_layout()
        
    print (label)
    # Сохраняем график
    output_file = os.path.join("all_dataframes_plots/all_in_one_plot_log_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранен в файл: {output_file}")
    plt.close()


    # Создаем интерактивный график
    plt.figure(figsize=(16, 8))

    # Цвета для разных линий
    colors = plt.cm.tab20(np.linspace(0, 1, 13))
    # Обрабатываем каждый файл 
    for i, csv_file in enumerate(csv_files): 
        try: 
            # Читаем CSV файл 
            df = pd.read_csv(directory_path + csv_file.split(".")[0] + "_corrected.csv") 
            print(directory_path + csv_file.split(".")[0] + "_corrected.csv")
            # Получаем имя для легенды
            if i < len(original_files):
                label = original_files[i]
            else:
                label = f"Dataset {i}"

            # Необязательные погрешности
            xerr = df['dEa (eV)'] if 'dEa (eV)' in df.columns else None
            yerr = df['dXS (b)']  if 'dXS (b)'  in df.columns else None
            
            # Строим график
            if label == "JENDL":
                plt.plot(df['Ea (eV)'], df['XS (b)'], "-", 
                        color=colors[i], linewidth=5, label=label, alpha=0.3)
            else:
                plt.errorbar(df['Ea (eV)'], df['XS (b)'], 
                             xerr=xerr,
                             yerr=yerr,
                            #  fmt='.-',
                            fmt='.',
                             capsize=3,
                             elinewidth=1,
                             markersize=4,
                             color=colors[i], 
                            #  linewidth=2, 
                             label= label, 
                             alpha=0.8
                        )

            print(f"На график добавлен файл: {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"Ошибка при обработке файла {csv_file}: {e}")
    
    # Настройки графика
    plt.xlabel('Alpha Energy (eV)', fontsize=14)
    plt.ylabel('Cross Section (b)', fontsize=14)
    plt.title('Cross Sections for All Datasets', fontsize=16)
    plt.legend(fontsize=12)

    plt.xlim(0, 8e6)
    plt.tick_params(axis='both')
    plt.grid(visible=True, alpha=0.3, which='both', axis='both')
    # Улучшаем оформление
    plt.tight_layout()

    print (label)
    # Сохраняем график
    output_file = os.path.join("all_dataframes_plots/all_in_one_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранен в файл: {output_file}")

    # plt.close()
    # Показываем график
    plt.show()


def plot_2_dataframes(directory_path):
    """
    Строит график для всех интерполированных датафреймов на одном графике
    
    Параметры:
    directory_path: путь к директории с интерполированными CSV файлами
    """
    # Находим все корректные CSV файлы в директории
    csv_files = [
        '1000_JENDL.csv',
        '100_Mohr.csv', 
        '2_Bair_1973.csv', 
        '7_Sekharan_1967.csv',
        '6_Brandenburg_2023.csv', 

        '5_Walton_1957.csv', 
        '10_Prusachenko_2022.csv',
        '4_Febbraro_2020.csv', 

        '1_Drotleff_1993.csv', 
        '3_Kellogg_1989.csv', 
        '8_Davids_1968.csv',
        # '8_Harissopulos_2005_needs_correction.csv',
        '11_Gao_2022.csv'
    ]

    # Имена исходных файлов для легенды
    original_files = [# csv_files
        "JENDL", 
        "S.Harissopulos & P.Mohr",
        "J.K.Bair et.al.", 
        "K.K.Sekharan et.al.", 
        "K. Brandenburg et.al.", # надо перевести в ЛСО из СЦМ

        "R.B.Walton et.al.", 
        "P.S.Prusachenko et.al.", 
        "M. Febbraro et.al.",
        
        "H.W.Drotleff et.al.", # надо перевести в ЛСО из СЦМ
        "S.E.Kellogg et.al.", 
        "C.N.Davids et.al.", 
        # "S.Harissopulos et.al.", # нет поправки Питера Мора, исключаю из рассмотрения
        "B.Gao et.al."
    ]

    colors = plt.cm.tab20(np.linspace(0, 1, 13))
    
    # Сортируем файлы по номеру
    # csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if len(csv_files) == 0:
        print("Не найдено корректных CSV файлов")
        return
    
    # Обрабатываем каждый файл 
    for i, csv_file in enumerate(csv_files): 
        try: 
            # Читаем CSV файл 
            df = pd.read_csv(directory_path + csv_file.split(".")[0] + "_corrected.csv") 
            print(directory_path + csv_file.split(".")[0] + "_corrected.csv")
            
            # Получаем имя для легенды
            if i < len(original_files):
                label = original_files[i]
            else:
                label = f"Dataset {i}"
            
            # Строим график
            plt.figure(figsize=(15, 6))

            if label == "JENDL":
                plt.plot(df['Ea (eV)'], df['XS (b)'], "-", 
                        color=colors[i], linewidth=5, label=label, alpha=0.3)
                JENDL_EN = df['Ea (eV)']
                JENDL_XS = df['XS (b)']
            elif label == "P.S.Prusachenko et.al." or label == "M. Febbraro et.al.":
                plt.plot(df['Ea (eV)'], df['XS (b)'], ".-", 
                        color=colors[i], linewidth=2, label=r"$(\alpha, n_0)$ " + label, alpha=1)
                plt.plot(JENDL_EN, JENDL_XS, "-", 
                        color=colors[0], linewidth=5, label="JENDL", alpha=0.3)
            else:
                plt.plot(df['Ea (eV)'], df['XS (b)'], ".-", 
                        color=colors[i], linewidth=2, label=label, alpha=1)
                plt.plot(JENDL_EN, JENDL_XS, "-", 
                        color=colors[0], linewidth=5, label="JENDL", alpha=0.3)
            
            print(f"На график добавлен файл: {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"Ошибка при обработке файла {csv_file}: {e}")
    
        # Настройки графика
        plt.xlabel('Alpha Energy (eV)', fontsize=14)
        plt.ylabel('Cross Section (b)', fontsize=14)
        plt.title('Cross Sections for All Datasets', fontsize=16)
        plt.legend(fontsize=12)
        
        # Логарифмическая шкала по Y для лучшего отображения
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlim(0, np.min([1.5*np.max(df['Ea (eV)']), 8e6]))
        plt.ylim(0, 1.5*np.max(df['XS (b)']))
        plt.tick_params(axis='both')
        plt.grid(visible=True, alpha=0.3, which='both', axis='both')
        # Улучшаем оформление
        plt.tight_layout()
        
        print (label)
        # Сохраняем график
        output_file = os.path.join("all_dataframes_plots/" + label + "_plot.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nГрафик сохранен в файл: {output_file}")
            
        plt.close()


def main():
    # Путь к директории с данными
    data_directory = "./full_data_corrected/"
    
    # Строим график для всех датафреймов
    plot_2_dataframes(data_directory)
    plot_all_dataframes(data_directory)


if __name__ == "__main__":
    main()