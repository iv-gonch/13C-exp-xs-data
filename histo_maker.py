import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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

    plt.savefig("./histo_data/" + fname + ".png")
    plt.close()
    plt.show()


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


fnames = [  
    '5_Brandenburg_2023', 
    '0_Drotleff_1993', 
    '1_Bair_1973', 
    '2_Kellogg_1989', 
    '3_Febbraro_2020', 
    '4_Walton_1957', 
    '6_Sekharan_1967', 
    '7_Davids_1968', 
    '9_Prusachenko_2022', 
    '10_Gao_2022', 
    '100_Mohr'
]
for fname in fnames:
    # input_csv = "./full_data_corrected/" + fname + "_corrected.csv"
    # # читаем данные
    # df = pd.read_csv(input_csv)

    # histo_make(df, n_bins=50)
    histo_plot(fname, n_bins=50)