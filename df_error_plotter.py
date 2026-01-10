import pandas as pd
import matplotlib.pyplot as plt

def plot_xs_from_csv(name):
    # Чтение данных
    df = pd.read_csv('./full_data_corrected/' + name + '_corrected.csv')

    # Обязательные столбцы
    if 'XS (b)' not in df.columns or 'Ea (eV)' not in df.columns:
        raise ValueError("В файле должны быть столбцы 'XS (b)' и 'Ea (eV)'")

    x = df['Ea (eV)']
    y = df['XS (b)']

    # Необязательные погрешности
    xerr = df['dEa (eV)'] if 'dEa (eV)' in df.columns else None
    yerr = df['dXS (b)']  if 'dXS (b)'  in df.columns else None

    # Построение графика
    plt.figure(figsize=(7, 5), dpi=200)

    plt.errorbar(
        x, y,
        xerr=xerr,
        yerr=yerr,
        fmt='.',
        capsize=3,
        elinewidth=1,
        markersize=4
    )

    title = name + ' XS(Ea)' 
    if not 'dXS (b)' in df.columns:
        title += ' no XS errors'
    if not 'dEa (eV)' in df.columns: 
        title += ' no Ea errors'
    
    plt.xlabel('Ea (eV)')
    plt.ylabel('XS (b)')
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./error_plots/' + name + '.png')


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
    '100_Mohr']

for name in fnames:
    plot_xs_from_csv(name)