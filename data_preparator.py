# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

M_C_13 = 12.89165 # а.е.м. масса атома углерода 13 
M_He_4 = 3.968219 # а.е.м. масса атома гелия 4
print("Атом He4", ((M_C_13)/(M_C_13+M_He_4))**2)
# M_He_4 = 4.002603 # а.е.м. масса ядра гелия 4
# print("Ядро He4", ((M_C_13)/(M_C_13+M_He_4))**2)

def data_preparator(directory_path):
    
    fname = '1_Drotleff_1993'
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    # Переименование нескольких столбцов одновременно
    df = df.rename(columns={
        'DATA (B) 0.1'      : 'XS (b)',
        'ERR-T (B) 0.911'   : 'dXS (b)',
        'EN-CM (EV) 1.1'    : 'Ea (eV)',
    })
    # df['Ea (eV)'] *= ((M_C_13)/(M_C_13+M_He_4))**2
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)

    # ================ #
    
    fname = '2_Bair_1973_needs_XSx0,8'
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    # Переименование нескольких столбцов одновременно
    df = df.rename(columns={
        'DATA (B) 0.1'              : 'XS (b)',
        'DATA-ERR (B) 0.911'        : 'dXS (b)',
        'EN (EV) 1.1'               : 'Ea (eV)',
        'EN-ERR (PER-CENT) 1.911'   : 'dEa (%)'
    })

    df['XS (b)'] *= 0.8 
    df['dXS (b)'] *= 0.8 
    df['dEa (eV)'] = df['Ea (eV)'] * df['dEa (%)']/100
    # COMMENT    By Gerry Hale (2014-10-16) and compiler (2014-10-20):
    # Correction factor 0.80 to 0.85 mentioned in the
    # 'Note added in proof' of Phys.Rev.C7(1973)1356 is not
    # applied to the data set compiled in this EXFOR entry. 
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)', 'dEa (eV)']]
    df.to_csv('./full_data_corrected/1_Bair_1973_corrected.csv', 
              float_format='%.15e', index=False)
    
    # ================ #

    fname = '3_Kellogg_1989'

    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'      : 'XS (b)',
        'ERR-S (B) 0.944'   : 'dXS (b)',
        'EN (EV) 1.1'       : 'Ea (eV)'
    })
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)
    
    # ================ #
    
    fname = '4_Febbraro_2020'

    df = pd.read_csv('./full_data_stage_1/' + fname + '_partial_alpha_n_0.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'      : 'XS (b)',
        'DATA-ERR (B) 0.911': 'dXS (b)',
        'EN (EV) 1.1'       : 'Ea (eV)'
    })
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)
    
    # ================ #
    
    fname = '5_Walton_1957'
    
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'  : 'XS (b)',
        'EN (EV) 1.1'   : 'Ea (eV)'
    })
    df = df[['XS (b)', 'Ea (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)

    # ================ #
    
    fname = '6_Brandenburg_2023'
    
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'              : 'XS (b)',
        'ERR-S (B) 0.944'           : 'dXS st (b)', # Statistical uncertainty.
        'ERR-1 (PER-CENT) 0.955'    : 'dXS 1 (%)',  # Neutron detection efficiency.
        'ERR-2 (PER-CENT) 0.955'    : 'dXS 2 (%)',  # Target thickness.
        'ERR-3 (PER-CENT) 0.955'    : 'dXS 3 (%)',  # Integrated beam current.
        'ERR-SYS (PER-CENT) 0.955'  : 'dXS sy (%)', # Systematic uncertainty.
        'EN-CM (EV) 1.1'            : 'Ea (eV)',
        'EN-RSL (PER-CENT) 1.922'   : 'dEa (%)'    # resolution
    })
    df['Ea (eV)'] *= ((M_C_13)/(M_C_13+M_He_4))**2  # CM to LAB

    df['dEa (eV)'] = df['Ea (eV)'] * df['dEa (%)']/100. 
    df['dXS (b)'] = df['XS (b)'] * np.sqrt(
        0.16**2. +  # Total uncertainty = 16% (see paper "Brandenburg K., Measurements of the 13C(alpha, n)16O cross section up to E_alpha = 8 MeV, 2023") 
        (df['dXS st (b)']/df['XS (b)'])**2. 
    )
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)', 'dEa (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)

    # ================ #
        
    fname = '7_Sekharan_1967'
    
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'          : 'XS (b)',     # (legacy) The absolute error in the cross section is estimated to be 20%
        'ERR-1 (PER-CENT) 0.955': 'dXS 1 (%)',  # The error in counter efficiency (12%)
        'ERR-2 (PER-CENT) 0.955': 'dXS 2 (%)',  # Error in current integrator (2%)
        'ERR-DIG (B) 0.955'    : 'dXS 3 (b)',  # Digitizing error
        'EN (EV) 1.1'           : 'Ea (eV)',
        'EN-ERR-DIG (EV) 1.955' : 'dEa (eV)'    # Digitizing error
    }) 

    df['dXS (b)'] = df['XS (b)'] * np.sqrt(
        (df['dXS 1 (%)']/100.)**2. + 
        (df['dXS 2 (%)']/100.)**2. + 
        (df['dXS 3 (b)']/df['XS (b)'])**2.
    )
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)', 'dEa (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)

    # ================ #
        
    fname = '8_Davids_1968'
    
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'          : 'XS (b)', 
        'ERR-1 (PER-CENT) 0.955': 'dXS 1 (%)',  # Error in the beam current integration 
        'ERR-2 (PER-CENT) 0.955': 'dXS 2 (%)',  # Error in the stopping cross section 
        'ERR-3 (PER-CENT) 0.955': 'dXS 3 (%)',  # Error in the geometry and in the lower level 
        'EN (EV) 1.1'           : 'Ea (eV)' 
    })
    df['dXS (b)'] = df['XS (b)'] * np.sqrt(
        (df['dXS 1 (%)']/100.)**2. + 
        (df['dXS 2 (%)']/100.)**2. + 
        (df['dXS 3 (%)']/100.)**2.
    )
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)

    # ================ #
        
    fname = '10_Prusachenko_2022'
    
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'              : 'XS (b)',
        'ERR-S (PER-CENT) 0.944'    : 'dXS s (%)',  # Statistical uncertainty 
        'ERR-SYS (PER-CENT) 0.955'  : 'dXS sys (%)',# Systematic uncertainty 
        'ERR-1 (PER-CENT) 0.955'    : 'dXS 1 (%)',  # Target thickness 
        'ERR-2 (PER-CENT) 0.955'    : 'dXS 2 (%)',  # Detector efficiency 
        'ERR-3 (PER-CENT) 0.955'    : 'dXS 3 (%)',  # Beam current 
        'ERR-4 (PER-CENT) 0.955'    : 'dXS 4 (%)',  # Solid Angle 
        'ERR-5 (PER-CENT) 0.955'    : 'dXS 5 (%)',  # Multiple scattering correction 
        'EN (EV) 1.1'               : 'Ea (eV)',
        'EN-RSL-FW (EV) 1.922'      : 'dEa (eV)'    # EN-RSL-FW is a total energy resolution based on total loss in a target.
    })
    df['dXS (b)'] = df['XS (b)'] * np.sqrt(
        (df['dXS s (%)']/100.)**2. + 
        (df['dXS sys (%)']/100.)**2. + 
        (df['dXS 1 (%)']/100.)**2. + 
        (df['dXS 2 (%)']/100.)**2. + 
        (df['dXS 3 (%)']/100.)**2. + 
        (df['dXS 4 (%)']/100.)**2. + 
        (df['dXS 5 (%)']/100.)**2. 
    )
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)', 'dEa (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)

    # ================ #
        
    fname = '11_Gao_2022_inverse_n_alpha'
    
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'          : 'XS (b)',
        'ERR-T (B) 0.911'       : 'dXS (b)',    # Include statistical uncertainties and systematic (estimated to be 11%) uncertainties
        'ERR-1 (PER-CENT) 0.955': 'dXS 1 (%)',  # Contributions from the beam current integration
        'ERR-2 (PER-CENT) 0.955': 'dXS 2 (%)',  # Detection efficiency
        'ERR-3 (PER-CENT) 0.955': 'dXS 3 (%)',  # Stopping power
        'ERR-4 (PER-CENT) 0.955': 'dXS 4 (%)',  # Angular distribution
        'EN (EV) 1.1'           : 'Ea (eV)'
    })
    df = df[['XS (b)', 'dXS (b)', 'Ea (eV)']] 
    df.to_csv('./full_data_corrected/10_Gao_2022_corrected.csv', 
              float_format='%.15e', index=False)

    # ================ #
        
    fname = '100_Mohr'
    
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'  : 'XS (b)',
        'EN (EV) 1.1'   : 'Ea (eV)'
    })
    df = df[['XS (b)', 'Ea (eV)']]
    df['XS (b)'] *= 1e-3
    df['Ea (eV)']*= 1e6
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)


    fname = '1000_JENDL'
    
    df = pd.read_csv('./full_data_stage_1/' + fname + '.csv')
    df = df.rename(columns={
        'DATA (B) 0.1'  : 'XS (b)',
        'EN (EV) 1.1'   : 'Ea (eV)'
    })
    df = df[['XS (b)', 'Ea (eV)']]
    df.to_csv('./full_data_corrected/' + fname + '_corrected.csv', 
              float_format='%.15e', index=False)


def main():
    # Путь к директории с данными
    data_directory = '.'
    
    # Строим график для всех датафреймов
    data_preparator(data_directory)

if __name__ == '__main__':
    main()