import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
from io import StringIO


def histo_plot(fname, n_bins = 50):
    input_csv = "./full_data_corrected/" + fname + "_corrected.csv"
    # читаем данные
    df = pd.read_csv(input_csv)

    # plt.hist(df["XS (b)"], bins=n_bins)
    plt.hist(df["Ea (eV)"]/1e6, bins = 75)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel("Ea (MeV)")
    plt.ylabel("Number of meashurements in energy bin")
    # plt.xlim(0, 15e6)
    plt.title("Histogram of " + fname)

    plt.savefig("./histo_plot/" + fname + "_frequency.png")
    plt.close()


def histo_make(df, n_bins = 50):
    # # переименуем для удобства
    # df = df.rename(columns={
    #     "XS (b)": "XS",
    #     "dXS (b)": "dXS",
    #     "Ea (eV)": "Ea",
    #     "dEa (eV)": "dEa"
    # })

    # считаем XS*Ea
    df["XS_Ea (b*eV)"] = df["XS (b)"] * df["Ea (eV)"]

    # погрешность XS*Ea
    df["dXS_Ea (b*eV)"] = np.sqrt(
        (df["Ea (eV)"] * df["dXS (b)"])**2 +
        (df["XS (b)"] * df["dEa (eV)"])**2
    )

    # строим бины по энергии
    bins = np.linspace(df["Ea (eV)"].min(), df["Ea (eV)"].max(), n_bins + 1)
    df["Ea_bin"] = pd.cut(df["Ea (eV)"], bins=bins)

    # агрегация по бинам
    hist = df.groupby("Ea_bin").agg(
        XS_Ea_sum=("XS_Ea (b*eV)", "sum"),
        dXS_Ea=("dXS_Ea (b*eV)", lambda x: np.sqrt(np.sum(x**2))),
        N_Ea=("XS_Ea (b*eV)", "count"),
        Ea_mean=("Ea (eV)", "mean")
    ).reset_index(drop=True)

    # # финальные имена столбцов
    # hist = hist.rename(columns={
    #     "XS_Ea_sum": "XS*Ea (b*eV)",
    #     "dXS_Ea": "d(XS*Ea) (b*eV)",
    #     "N_Ea": "N_Ea",
    #     "Ea_mean": "Ea (eV)"
    # })

    # сохраняем
    output_csv = "./histo_data/" + fname + "_histo.csv" 
    hist.to_csv(output_csv, index=False) 

    print("Гистограмма сохранена в", output_csv) 


def gauss_histo_plot(fname): 

    input_csv = "./full_data_corrected/" + fname + "_corrected.csv" 
    df = pd.read_csv(input_csv) 

    # Необязательные погрешности 
    xerr = df['dEa (eV)'] if 'dEa (eV)' in df.columns else np.zeros_like(df['Ea (eV)']) 
    yerr = df['dXS (b)']  if 'dXS (b)'  in df.columns else np.zeros_like(df['XS (b)'])

    df.drop(['dEa (eV)'], axis=1, inplace=True) if 'dEa (eV)' in df.columns else None 
    df.drop(['dXS (b)'],  axis=1, inplace=True) if 'dXS (b)'  in df.columns else None 

    df.insert(1, "d_sigma", yerr) 
    df.insert(3, "dE", xerr) 

    # Сначала переименовываем существующие колонки (игнорируя отсутствующие через errors='ignore')
    rename_dict = {
        'XS (b)': 'sigma',
        'dXS (b)': 'd_sigma',
        'Ea (eV)': 'E',
        'dEa (eV)': 'dE'
    }
    df = df.rename(columns=rename_dict, errors='ignore')

    # Если отсутствует d_sigma или dE, создаём их как нулевые колонки (дефолт погрешность=0)
    if 'd_sigma' not in df.columns:
        df['d_sigma'] = 0.0
    if 'dE' not in df.columns:
        df['dE'] = 0.0

    # Проверяем наличие обязательных колонок (sigma и E), иначе ошибка
    if 'sigma' not in df.columns or 'E' not in df.columns:
        raise ValueError("Обязательные колонки 'XS (b)' и 'Ea (eV)' отсутствуют в данных")

    df = df.sort_values('E')

    delta_E_desired = 2_000 # eV 

    # Диапазон и бины 
    E_min, E_max = 300_000, 8_010_000   # eV 

    num_bins = int(np.ceil((E_max - E_min) / delta_E_desired)) or 1 
    bin_edges = np.linspace(E_min, E_max, num_bins + 1) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    delta_E = np.diff(bin_edges) 

    # Fractional binning 
    binned_sigma = np.zeros(num_bins) 
    binned_d_sigma_sq = np.zeros(num_bins) 
    weights = np.zeros(num_bins) 

    has_dE = 'dE' in df.columns and df['dE'].notna().all() and (df['dE'] > 0).any()  # Глобальная проверка: есть ли колонка dE и ненулевые значения

    for i, row in df.iterrows():
        if has_dE and row['dE'] > 0:  # Если есть dE и для этой строки >0, используем fractional
            fractions = norm.cdf(bin_edges[1:], row['E'], row['dE']) - norm.cdf(bin_edges[:-1], row['E'], row['dE'])
        else:  # Иначе (dE нет или =0) — жесткий биннинг: 1 в бине с E, 0 в остальных
            fractions = np.zeros(num_bins)
            bin_idx = np.digitize(row['E'], bin_edges) - 1  # Находим индекс бина (0-based)
            if 0 <= bin_idx < num_bins:
                fractions[bin_idx] = 1.0
        
        mask = fractions > 1e-6 # булиевый массив такой же длинны, что и fractions
        binned_sigma[mask] += row['sigma'] * fractions[mask]    # добавляем вклад текущей точки только в бины, где mask=True 
        binned_d_sigma_sq[mask] += (row['d_sigma'] * fractions[mask]) ** 2  # дисперсия суммы в предположении независимости сл.величин
        weights[mask] += fractions[mask]    # сумма весов в бине от каждой точки 

    mask_nonzero = weights > 0 # булиевый массив длинны num_bins 
    binned_sigma[mask_nonzero] /= weights[mask_nonzero] # сечение в бине обратно пропорционально сумме весов в бине 
    binned_d_sigma = np.sqrt(binned_d_sigma_sq) # корень из суммы квадратов погрешностей с учётом весов
    binned_d_sigma[mask_nonzero] /= weights[mask_nonzero]   # делим на вес, чтобы получить среднее. Тк сечение - это усреднённое интегральное значение

    # Y = sigma * delta_E 
    y = binned_sigma * delta_E 
    err_y = binned_d_sigma * delta_E 
    df_histo = pd.DataFrame({'XS_Ea_sum': y, 'dXS_Ea': err_y, 'Ea_mean': bin_centers}) 
    # print(df_histo.head()) 
    df_histo.to_csv('./histo_data/' + fname + '.csv', index=False)

    # График 
    plt.figure(figsize=(10, 6)) 

    # plt.step(bin_edges, np.concatenate([y[0:1] + err_y[0:1], y + err_y]),        where='pre', color='b', alpha=0.5)
    plt.fill_between(bin_edges, np.concatenate([y[0:1] + err_y[0:1], y + err_y]), step="pre", color='b', alpha=0.5)
    plt.step(bin_edges, np.concatenate([y[0:1], y]),                             where='pre', color='b', label='sigma * ΔE') 
    plt.fill_between(bin_edges, np.concatenate([y[0:1] - err_y[0:1], y - err_y]), step="pre", color='w', alpha=0.7)
    plt.step(bin_edges, np.concatenate([y[0:1] - err_y[0:1], y - err_y]),        where='pre', color='b', alpha=0.5) 
    # plt.errorbar(bin_centers[mask_nonzero], y[mask_nonzero], xerr=delta_E[mask_nonzero]/2, yerr=err_y[mask_nonzero], fmt='none', ecolor='r', capsize=3, label='Errors') 
    plt.xlabel('E (eV)') 
    plt.ylabel('sigma * ΔE') 
    plt.title(f'Ступенчатый график с fractional binning ΔE = {delta_E_desired/1e3} keV') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig('./histo_plot/' + fname + '.png') 
    # plt.show()


fnames = [
    '1_Drotleff_1993', 
    '2_Bair_1973', 
    '3_Kellogg_1989', 
    '4_Febbraro_2020', 
    '5_Walton_1957', 
    '6_Brandenburg_2023', 
    '7_Sekharan_1967', 
    '8_Davids_1968', 
    '10_Prusachenko_2022', 
    '11_Gao_2022', 
    '100_Mohr', 
    '1000_JENDL' 
]

for fname in fnames: 
    # input_csv = "./full_data_corrected/" + fname + "_corrected.csv" 
    # # читаем данные 
    # df = pd.read_csv(input_csv) 

    # histo_plot(fname, n_bins=50) 
    gauss_histo_plot(fname)  