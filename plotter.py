import pandas as pd
import matplotlib.pyplot as plt

def plot_xs_from_csv(name):
    # Чтение данных
    df = pd.read_csv('./full_data_corrected/' + name +'_corrected.csv')

    # Обязательные столбцы
    if 'XS (b)' not in df.columns or 'Ea (eV)' not in df.columns:
        raise ValueError("В файле должны быть столбцы 'XS (b)' и 'Ea (eV)'")

    x = df['Ea (eV)']/1e6
    y = df['XS (b)']

    # Необязательные погрешности
    xerr = df['dEa (eV)']/1e6 if 'dEa (eV)' in df.columns else None
    yerr = df['dXS (b)'] if 'dXS (b)' in df.columns else None

    # Построение графика
    plt.figure(figsize=(7, 5), dpi=200)

    plt.errorbar(
        x, y,
        xerr=xerr,
        yerr=yerr,
        fmt='o',
        capsize=3,
        elinewidth=1,
        markersize=4
    )

    plt.xlabel('Ea (MeV)')
    plt.ylabel('XS (b)')
    plt.title(name + ' XS(Ea)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./error_plots/' + name + '.png')


fnames = [
    '0_Drotleff_1993',
    '1_Bair_1973',
    '2_Kellogg_1989',
    '3_Febbraro_2020',
    '4_Walton_1957',
    '5_Brandenburg_2023',
    '6_Sekharan_1967',
    '7_Davids_1968',
    '9_Prusachenko_2022',
    '10_Gao_2022',
    '100_Mohr']

for name in fnames:
    plot_xs_from_csv(name)