import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ternary
import zipfile

from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages  # (mantido)

plt.switch_backend('Agg')


def calculate_bounds2(download_dir):
    """
    Lê SoilSorted.xlsx e calcula:
      - df_iniciais (code, T15000, alpha)  [ANTES: InitialParam.txt]
      - df_bounds (Med0, Med15000, Min0, Max0, Max15000) [ANTES: Bounds.xlsx]
    Retorna tudo em memória (DataFrames), sem salvar arquivos intermediários.
    """
    df_soil_sorted = pd.read_excel(os.path.join(download_dir, 'SoilSorted.xlsx'))

    # Calculate the alpha and T15000 values
    codes = []
    t15000_values = []
    alpha_values = []

    for code in df_soil_sorted['code'].unique():
        df_temp = df_soil_sorted[df_soil_sorted['code'] == code].reset_index(drop=True)

        # Mantém a mesma lógica original (dependente de ordenação)
        third_theta = df_temp.iloc[2]['theta']
        fourth_theta = df_temp.iloc[3]['theta']
        first_theta = df_temp.iloc[0]['theta']
        h_third_line = df_temp.iloc[2]['h']

        alpha = (
            (((third_theta - (fourth_theta - 0.01)) / (first_theta - (fourth_theta - 0.01))) ** (-1 / 0.5) - 1)
            ** (1 - 0.5)
        ) / h_third_line
        t15000_temp = df_temp.iloc[3]['theta'] - 0.01

        codes.append(code)
        t15000_values.append(t15000_temp)
        alpha_values.append(alpha)

    df_iniciais = pd.DataFrame({'code': codes, 'T15000': t15000_values, 'alpha': alpha_values})

    # Calculate bounds (mesma lógica do original)
    med0 = df_soil_sorted[df_soil_sorted['h'] == 0]['theta'].mean()
    med15000 = df_soil_sorted[df_soil_sorted['h'] >= 9000]['theta'].mean() - 0.01
    min0 = df_soil_sorted[df_soil_sorted['h'] == 0]['theta'].min()
    max0 = df_soil_sorted[df_soil_sorted['h'] == 0]['theta'].max()
    max15000 = df_soil_sorted[df_soil_sorted['h'] >= 9000]['theta'].max()

    df_bounds = pd.DataFrame({
        'Med0': [med0],
        'Med15000': [med15000],
        'Min0': [min0],
        'Max0': [max0],
        'Max15000': [max15000]
    })

    return df_soil_sorted, df_bounds, df_iniciais


def calculate_parameters2(download_dir, plots_dir, df_soil_sorted, df_bounds, df_iniciais):
    """
    Executa todo o pipeline mantendo intermediários como DataFrames em memória,
    SEM salvar Excels intermediários (Parameters/RMSE/ERRORMAX/Adherence/SoilSorted_Head/Bounds/InitialParam/ret).

    Mantém as plotagens e salvamentos de imagens/ZIP NOS MESMOS LOCAIS do código original.

    Salva apenas:
      - YourSoilClassified.xlsx (no download_dir)
    """

    # ----------------------------
    # Ajuste Van Genuchten (mesma forma, mas sem ret.txt / sem leitura de Parameters.xlsx etc.)
    # ----------------------------

    def vg4(par, h, par1):
        return np.abs(par[0]) + (par1 - np.abs(par[0])) / (1 + (par[1] * h) ** par[2]) ** (1 - 1 / par[2])

    def vg4ssq(par, press, theta, par1):
        theta1 = vg4(par, press, par1)
        diff = theta - theta1
        ssq = np.dot(diff, diff)
        return ssq

    def vg4ssq2(press, par0, par2, par3, par1):
        par = [par0, par2, par3]
        return vg4(par, press, par1)

    # Bounds
    med15000_value = df_bounds['Med15000'].iloc[0]
    med0_value = df_bounds['Med0'].iloc[0]
    min0_value = df_bounds['Min0'].iloc[0]
    max15000_value = df_bounds['Max15000'].iloc[0]
    max0_value = df_bounds['Max0'].iloc[0]

    unique_codes = list(df_soil_sorted['code'].unique())
    MAXSAMP = len(unique_codes)

    # Arrays resultados (ThetaR, ThetaS, alpha, n)
    res2 = np.zeros((MAXSAMP, 4))
    rmse2 = np.zeros(MAXSAMP)

    # Ajuste por code usando 3 pontos (h/theta das linhas 2,3,4 do grupo; ThetaS fixo = theta do h=0 / primeira linha)
    for i, code in enumerate(unique_codes):
        group = df_soil_sorted[df_soil_sorted['code'] == code].reset_index(drop=True)

        if len(group) >= 4:
            # Pega 3 pontos: (iloc 1,2,3) e ThetaS fixo = theta do iloc 0 (mesma intenção do original)
            press = group.loc[[1, 2, 3], 'h'].to_numpy(dtype=float)
            theta = group.loc[[1, 2, 3], 'theta'].to_numpy(dtype=float)

            par1_value = float(group.loc[0, 'theta'])  # ThetaS (fixo)

            # Iniciais
            row_ini = df_iniciais[df_iniciais['code'] == code]
            if row_ini.empty:
                t15000_value = med15000_value
                alpha_value = 1.0
            else:
                t15000_value = float(row_ini['T15000'].values[0])
                alpha_value = float(row_ini['alpha'].values[0])

            # Ensure within bounds
            t15000_value = np.clip(t15000_value, 0, max15000_value)
            alpha_value = np.clip(alpha_value, 0.001, 10000)

            par0_init = t15000_value
            par2_init = alpha_value
            par3_init = 2.0
            p0 = [par0_init, par2_init, par3_init]

            bounds = ([0, 0.001, 1], [max15000_value, 10000, 10])

            try:
                res2a, _ = curve_fit(
                    lambda press, par0, par2, par3: vg4ssq2(press, par0, par2, par3, par1_value),
                    press,
                    theta,
                    p0=p0,
                    bounds=bounds,
                    method='trf',
                    maxfev=100000,
                    absolute_sigma=True
                )

                ssq2a = vg4ssq(res2a, press, theta, par1_value)

                res2[i, 0] = res2a[0]       # ThetaR
                res2[i, 1] = par1_value     # ThetaS
                res2[i, 2] = res2a[1]       # alpha
                res2[i, 3] = res2a[2]       # n
                rmse2[i] = np.sqrt(ssq2a / 3.0)

            except Exception:
                res2[i, :] = -9.9
                rmse2[i] = -9.9

        else:
            res2[i, :] = -9.9
            rmse2[i] = -9.9

    columns = ['ThetaR', 'ThetaS', 'alpha', 'n']
    df_parameters = pd.DataFrame(res2, columns=columns)
    df_parameters['m'] = 1 - 1 / df_parameters['n']
    df_parameters['RMSE'] = rmse2
    df_parameters.insert(0, 'code', unique_codes)

    # ----------------------------
    # SoilSorted_Head em memória (ANTES: SoilSorted_Head.xlsx)
    # ----------------------------
    df_tmp = df_soil_sorted.copy()
    if 'PT' in df_tmp.columns:
        df_tmp = df_tmp.drop(columns=['PT'])

    def transpose_h_theta(df):
        new_df = pd.DataFrame()
        for code in df['code'].unique():
            code_data = df[df['code'] == code].reset_index(drop=True)
            h_values = code_data['h']
            theta_values = code_data['theta']
            for j, (h, theta) in enumerate(zip(h_values, theta_values)):
                new_df.at[code, f'h{j}'] = h
                new_df.at[code, f'theta{j}'] = theta
                new_df.at[code, 'code'] = code
        return new_df

    df_head = transpose_h_theta(df_tmp)
    # Mantém exatamente as mesmas colunas do original
    df_head = df_head[['code', 'h0', 'theta0', 'h1', 'theta1', 'h2', 'theta2', 'h3', 'theta3']]

    # ----------------------------
    # RMSE total (em memória) [ANTES: RMSE.xlsx]
    # ----------------------------
    df_soil_complete = pd.read_excel(os.path.join(download_dir, 'SoilComplete.xlsx'))
    df_rmse_base = pd.merge(df_soil_complete, df_parameters, on='code')

    def calculate_theta_calculated(row):
        h = row['h']
        thetaR = row['ThetaR']
        thetaS = row['ThetaS']
        alpha = row['alpha']
        m = row['m']
        n = row['n']
        return thetaR + (thetaS - thetaR) / (1 + (alpha * h) ** n) ** (m)

    df_rmse_base['theta_calculated'] = df_rmse_base.apply(calculate_theta_calculated, axis=1)

    def calculate_RMSETOTAL(row):
        code = row['code']
        subset = df_rmse_base[df_rmse_base['code'] == code]
        num_lines = len(subset)
        divisor = (num_lines - 3) if num_lines > 3 else 3
        sum_squared_diff = np.sum((subset['theta'] - subset['theta_calculated']) ** 2)
        return np.sqrt(sum_squared_diff / divisor)

    df_rmse_base['TOTALRMSE'] = df_rmse_base.apply(calculate_RMSETOTAL, axis=1)

    # ----------------------------
    # ERRORMAX (em memória) [ANTES: ERRORMAX.xlsx]
    # ----------------------------
    df_soil = df_soil_sorted.copy()
    df_soil = df_soil[df_soil['h'] != 0]

    df_err_merged = pd.merge(df_soil, df_parameters, on='code', how='inner')

    df_err_merged['theta_calculated'] = (
        df_err_merged['ThetaR'] +
        (df_err_merged['ThetaS'] - df_err_merged['ThetaR']) /
        (1 + (df_err_merged['alpha'] * df_err_merged['h']) ** df_err_merged['n']) ** (df_err_merged['m'])
    )
    df_err_merged['difference'] = abs(df_err_merged['theta'] - df_err_merged['theta_calculated'])
    df_errormax = df_err_merged.groupby('code')['difference'].max().reset_index()
    df_errormax.rename(columns={'difference': 'ERRORMAX'}, inplace=True)

    # ----------------------------
    # Construção do df principal (equivalente ao que era "df" vindo de RMSE.xlsx + ERROMAX)
    # ----------------------------
    df = df_rmse_base.copy()
    df = pd.merge(df, df_errormax[['code', 'ERRORMAX']], on='code', how='left')

    # Colunas adicionais (mesma lógica original)
    df['W60'] = ((1 + (df['alpha'] * 60) ** df['n']) ** -df['m'])
    df['W15000'] = ((1 + (df['alpha'] * 15000) ** df['n']) ** -df['m'])
    df['A60'] = 1 - df['W60']
    df['W60-W15000'] = df['W60'] - df['W15000']
    df['W60%'] = df['W60'] * 100
    df['W15000%'] = df['W15000'] * 100
    df['A60%'] = 100 - df['W60%']
    df['W60%-W15000%'] = df['W60%'] - df['W15000%']

    # Remove h/theta/theta_calculated (mesmo)
    for col in ['h', 'theta', 'theta_calculated']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Mantém só a primeira linha por code
    df = df.groupby('code').first().reset_index()

    # Adherence e Classification (mesmo)
    df['Adherence'] = np.where(
        (df['ERRORMAX'] <= 0.02) & (df['TOTALRMSE'] <= 0.035), 'High',
        np.where((df['ERRORMAX'] > 0.04) | (df['TOTALRMSE'] > 0.07), 'Low', 'Medium')
    )

    df['Classification'] = np.where(
        df['Adherence'] == 'High', 'Genuine Soil',
        np.where(df['Adherence'] == 'Medium', 'Adopted Soil', 'Rejected Soil')
    )

    # Merge com df_head (ANTES: Adherence.xlsx era salvo)
    df = pd.merge(df_head, df, on='code', how='inner')

    # ----------------------------
    # Classificações adicionais (mesmo do original)
    # ----------------------------
    def classify_suborder(row):
        diff = row['theta0'] - row['ThetaR']
        if 0 < diff <= 0.20:
            return 1
        elif 0.20 < diff <= 0.40:
            return 2
        elif 0.40 < diff <= 0.60:
            return 3
        elif diff > 0.60:
            return 4
        else:
            return None

    def classify_order(row):
        if 2/3 <= row["A60"] < 1 and 0 < row['W60-W15000'] < 1/3 and 0 < row['W15000'] < 1/3:
            return "A"
        elif 1/3 <= row["A60"] < 2/3 and 0 < row['W60-W15000'] < 1/3 and 0 < row['W15000'] < 1/3:
            return "D"
        elif 1/3 <= row["A60"] < 2/3 and 1/3 <= row['W60-W15000'] < 2/3 and 0 < row['W15000'] < 1/3:
            return "B"
        elif 0 < row["A60"] < 1/3 and 2/3 <= row['W60-W15000'] < 1 and 0 < row['W15000'] < 1/3:
            return "C"
        elif 0 < row["A60"] < 1/3 and 1/3 <= row['W60-W15000'] < 2/3 and 0 < row['W15000'] < 1/3:
            return "E"
        elif 0 < row["A60"] < 1/3 and 1/3 <= row['W60-W15000'] < 2/3 and 1/3 <= row['W15000'] < 2/3:
            return "G"
        elif 0 < row["A60"] < 1/3 and 0 < row['W60-W15000'] < 1/3 and 2/3 <= row['W15000'] < 1:
            return "I"
        elif 0 < row["A60"] < 1/3 and 0 < row['W60-W15000'] < 1/3 and 1/3 <= row['W15000'] < 2/3:
            return "H"
        elif 1/3 <= row["A60"] < 2/3 and 0 < row['W60-W15000'] < 1/3 and 1/3 <= row['W15000'] < 2/3:
            return "F"
        else:
            return "Error"

    df["Order"] = df.apply(classify_order, axis=1)
    df["Suborder"] = df.apply(classify_suborder, axis=1)
    df["Suborder"] = df["Suborder"].fillna(0).astype(int)
    df["Family"] = df["Order"] + df["Suborder"].astype(str)

    suborder_dict = {
        1: "Low Effective Porosity",
        2: "Moderate Effective Porosity",
        3: "High Effective Porosity",
        4: "Very High Effective Porosity"
    }

    order_dict = {
        "A": "Highly Macrospacious Soil",
        "C": "Highly Mesospacious Soil",
        "I": "Highly Microspacious Soil",
        "D": "Macrospacious Soil",
        "E": "Mesospacious Soil",
        "H": "Microspacious Soil",
        "B": "Macro-Mesospacious Soil",
        "F": "Macro-Microspacious Soil",
        "G": "Meso-Microspacious Soil"
    }

    def get_family_nomenclature(row):
        porosity = suborder_dict.get(row['Suborder'], "Unknown Porosity")
        order_name = order_dict.get(row['Order'], "Unknown Soil Type")
        return f"{porosity} - {order_name}"

    df["Family Nomenclature"] = df.apply(get_family_nomenclature, axis=1)

    df['a (cm³/cm³)'] = df['theta0'] - df['theta2']
    df['w (cm³/cm³)'] = df['theta2'] - df['theta3']
    df['Ksat (cm/d)'] = 1931 * (df['a (cm³/cm³)'] ** 1.948)

    def classify_a_value(a):
        if a < 0.10:
            return "Low"
        elif 0.10 <= a <= 0.20:
            return "Moderate"
        elif a > 0.20:
            return "High"
        else:
            return "Undefined"

    df['a value'] = df['a (cm³/cm³)'].apply(classify_a_value)

    def classify_w_value(w):
        if w < 0.06:
            return "Low"
        elif 0.06 <= w <= 0.12:
            return "Moderate"
        elif w > 0.12:
            return "High"
        else:
            return "Undefined"

    df['w value'] = df['w (cm³/cm³)'].apply(classify_w_value)

    def classify_ksat_value(ksat):
        if ksat <= 12:
            return "Slow"
        elif 12 < ksat <= 120:
            return "Moderate"
        elif ksat > 120:
            return "Rapid"
        else:
            return "Undefined"

    df['ksat value'] = df['Ksat (cm/d)'].apply(classify_ksat_value)

    def classify_hydraulic_class(row):
        a_val = row['a value']
        w_val = row['w value']
        ksat_val = row['ksat value']

        if (a_val == "Low" and w_val in ["Low", "Moderate", "High"]):
            restriction = "Soil with air restriction"
        elif (a_val in ["High", "Moderate"] and w_val == "Low"):
            restriction = "Soil with water restriction"
        elif ((a_val == "Moderate" and w_val == "High") or
              (a_val == "High" and w_val in ["Moderate", "High"]) or
              (a_val == "Moderate" and w_val == "Moderate")):
            restriction = "Soil without air and water restriction"
        else:
            restriction = "Undefined Restriction"

        if ksat_val == "Slow":
            permeability = "slow permeability"
        elif ksat_val == "Moderate":
            permeability = "moderate permeability"
        elif ksat_val == "Rapid":
            permeability = "rapid permeability"
        else:
            permeability = "undefined permeability"

        return f"{restriction} and {permeability}"

    df['Hydraulic Class'] = df.apply(classify_hydraulic_class, axis=1)

    # Mesmo drop do original
    df.drop(columns=['a value', 'w value', 'ksat value', 'RMSE'], inplace=True)

    # Rename do TOTALRMSE
    df.rename(columns={"TOTALRMSE": "RMSE_30cm-18000cm"}, inplace=True)

    # Reorder para SampleID primeiro (mesmo)
    if 'SampleID' in df.columns:
        columns_order = ['SampleID'] + [col for col in df.columns if col != 'SampleID']
        df = df[columns_order]
        df['SampleID'] = df['SampleID'].astype(str)

    # ----------------------------
    # PLOTAGENS (NÃO ALTERADAS) — usa SoilFull.xlsx e df (que substitui adherence_df)
    # ----------------------------
    soilfull_df = pd.read_excel(os.path.join(download_dir, 'SoilFull.xlsx'))
    adherence_df = df.copy()  # substitui a leitura do Adherence.xlsx, mantendo o mesmo conteúdo em memória

    soilfull_df['log_h'] = np.log10(soilfull_df['h'].replace(0, np.nan))
    soilfull_df['Theta'] = soilfull_df['theta']

    unique_codes_plot = soilfull_df['code'].unique()

    os.makedirs(plots_dir, exist_ok=True)

    for code in unique_codes_plot:
        code_df = soilfull_df[soilfull_df['code'] == code]
        sample_id = code_df['SampleID'].values[0]

        plt.figure(figsize=(10, 6))
        plot_color = 'black'

        for index, row in code_df.iterrows():
            if row['h'] == 0:
                plt.scatter(0, row['Theta'], color=plot_color, marker='x', s=100, label='φ')
            else:
                plt.scatter(row['log_h'], row['Theta'], color=plot_color, alpha=0.6)

        adherence_row = adherence_df[adherence_df['code'] == code]
        if not adherence_row.empty:
            ThetaR = adherence_row['ThetaR'].values[0]
            ThetaS = adherence_row['ThetaS'].values[0]
            alpha = adherence_row['alpha'].values[0]
            n = adherence_row['n'].values[0]

            h_values_above_30 = np.linspace(30, soilfull_df['h'].max(), 100)
            Theta_values_above_30 = ThetaR + (ThetaS - ThetaR) / (1 + (alpha * h_values_above_30) ** n) ** (1 - 1 / n)

            plt.plot(np.log10(h_values_above_30), Theta_values_above_30, linestyle='--', color='gray')

            plt.scatter(np.log10(adherence_row['h1'].values[0]), adherence_row['theta1'].values[0],
                        color=plot_color, marker='s', s=100)
            plt.scatter(np.log10(adherence_row['h2'].values[0]), adherence_row['theta2'].values[0],
                        color=plot_color, marker='s', s=100)
            plt.scatter(np.log10(adherence_row['h3'].values[0]), adherence_row['theta3'].values[0],
                        color=plot_color, marker='s', s=100)

            error_max = round(adherence_row['ERRORMAX'].values[0], 3)
            total_rmse = round(adherence_row['RMSE_30cm-18000cm'].values[0], 3)

            handles = [
                plt.Line2D([0], [0], color='w', label=f'ERRORMAX: {error_max} cm³/cm³'),
                plt.Line2D([0], [0], color='w', label=f'RMSE$_{{30-18000}}$: {total_rmse} cm³/cm³'),
                plt.Line2D([0], [0], marker='s', color='black', label='θ1, θ2, θ3',
                           markerfacecolor='black', markersize=10, linestyle='None'),
                plt.Line2D([0], [0], marker='x', color='black', label='φ',
                           markersize=10, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='gray', label='θ',
                           markersize=6, linestyle='None')
            ]
            plt.legend(handles=handles, loc='upper right')

        plt.axhline(0, color='black', lw=0.8)
        plt.axvline(0, color='black', lw=0.8)
        plt.xlabel("log h (cm)")
        plt.ylabel("Theta (cm³/cm³)")
        plt.title(f"Retention curve - Soil {sample_id}")
        plt.grid()
        plt.tight_layout()
        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.savefig(os.path.join(plots_dir, f'retention_curve_code_{code}.jpg'))
        plt.close()

    def classify_suborder_for_plot(row):
        diff = row['theta0'] - row['ThetaR']
        if 0 < diff <= 0.20:
            return 1
        elif 0.20 < diff <= 0.40:
            return 2
        elif 0.40 < diff <= 0.60:
            return 3
        elif diff > 0.60:
            return 4

    def map_color(adherence):
        if adherence == 'High':
            return 'green'
        elif adherence == 'Medium':
            return '#FFC300'
        elif adherence == 'Low':
            return 'red'

    def plot_ternary(df_plot, title, filename):
        scale = 100
        try:
            figure, tax = ternary.figure(scale=scale)
        except AttributeError:
            raise Exception("A função 'figure' não está disponível no módulo 'ternary'. Verifique a instalação.")

        figure.set_size_inches(5, 5)

        step = scale / 9
        grid_color = '#a9a9a9'

        for i in range(9):
            for j in range(9 - i):
                x1, y1 = i * step, j * step
                x2, y2 = (i + 1) * step, j * step
                x3, y3 = i * step, (j + 1) * step

                tax.line((x1, y1), (x2, y2), color=grid_color, linestyle='--', linewidth=0.5)
                tax.line((x2, y2), (x3, y3), color=grid_color, linestyle='--', linewidth=0.5)
                tax.line((x3, y3), (x1, y1), color=grid_color, linestyle='--', linewidth=0.5)

        tax.boundary(linewidth=1.0)

        fontsize = 12
        offset = 0.2
        tax.left_axis_label("    W15000 (%)", fontsize=fontsize, offset=offset, rotation=-240)
        tax.right_axis_label("W60-W15000 (%)", fontsize=fontsize, offset=offset, rotation=240)
        tax.bottom_axis_label("      A60 (100-W60) (%)", fontsize=fontsize, offset=offset, rotation=180)

        tax.annotate("Microspace (%)", position=(62, 54), fontsize=10, rotation=60, va='center', ha='center')
        tax.annotate("Mesospace (%)", position=(-11.6, 54), fontsize=10, rotation=-60, va='center', ha='center')
        tax.annotate("Macrospace (%)", position=(54, -11.6), fontsize=10, rotation=0, va='center', ha='center')

        for index, row in df_plot.iterrows():
            color = map_color(row['Adherence'])
            tax.scatter([[row['A60%'], row['W15000%'], row['W60%-W15000%']]],
                        marker='o', s=35, color=color)

        plt.gca().invert_xaxis()
        triangles = [((66.7, 0), (66.7, 33.3)), ((33.3, 0), (33.3, 66.7)), ((66.7, 33.3), (0, 33.3)),
                     ((33.3, 66.7), (0, 66.7)), ((0, 66.7), (66.7, 0)), ((0, 33.3), (33.3, 0))]
        for p1, p2 in triangles:
            tax.line(p1, p2, linewidth=1, color='black')

        titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        midpoints = [(81, 9, 5), (47, 9.5), (13, 9.5), (59, 20), (24, 20), (45, 45), (13, 45), (26, 52), (12, 78)]
        for title_point, point in zip(titles, midpoints):
            tax.annotate(title_point, point)

        plt.title(title, pad=20)

        tax.ticks(axis='lbr', ticks=[0, 33.3, 66.7, 100], linewidth=1, offset=0.025)
        tax.get_axes().axis('off')
        tax.clear_matplotlib_ticks()

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Genuine Sample', markerfacecolor='green', markersize=5),
            plt.Line2D([0], [0], marker='o', color='w', label='Adopted Sample', markerfacecolor='#FFC300', markersize=5),
            plt.Line2D([0], [0], marker='o', color='w', label='Rejected Sample', markerfacecolor='red', markersize=5)
        ]
        plt.legend(handles=legend_elements, loc='upper left', title="Adherence", fontsize='small')

        plt.savefig(filename, format='jpg')
        plt.close()

    os.makedirs(plots_dir, exist_ok=True)

    for order in range(1, 5):
        filtered_df = df[df.apply(lambda row: classify_suborder_for_plot(row) == order, axis=1)]
        if not filtered_df.empty:
            plot_ternary(filtered_df, f"Sub-Order {order}", os.path.join(plots_dir, f'suborder_plot_1soil.jpg'))

    files_to_include = [
        'suborder_plot_1soil.jpg',
        'retention_curve_code_1.jpg'
    ]

    with zipfile.ZipFile(os.path.join(download_dir, 'ternary_plots.zip'), 'w') as zipf:
        for root, _, files in os.walk(plots_dir):
            for file in files:
                if file in files_to_include:
                    zipf.write(os.path.join(root, file), file)

    # ----------------------------
    # Renomes finais e salvamento ÚNICO do Excel final
    # ----------------------------
    df.rename(columns={
        "h0": "h0 (cm)",
        "theta0": "theta0 (cm³/cm³)",
        "h1": "h1 (cm)",
        "theta1": "theta1 (cm³/cm³)",
        "h2": "h2 (cm)",
        "theta2": "theta2 (cm³/cm³)",
        "h3": "h3 (cm)",
        "theta3": "theta3 (cm³/cm³)",
        "ThetaR": "ThetaR (cm³/cm³)",
        "ThetaS": "ThetaS (cm³/cm³)",
        "alpha": "alpha (cm-1)",
        "W15000%": "Microspace%",
        "A60%": "Macrospace%",
        "W60%-W15000%": "Mesospace%",
        "Classification": "Sample status",
        "Family Nomenclature": "Family Name"
    }, inplace=True)

    df.drop(columns=['Adherence', 'Order', 'Suborder', 'W60', 'W15000', 'A60', 'W60-W15000', 'W60%'], inplace=True)

    # ÚNICO Excel criado ao final:
    df.to_excel(os.path.join(download_dir, 'YourSoilClassified.xlsx'), index=False)

    return df




def process_excel(excel_file, download_dir, plots_dir):
    df = pd.read_excel(excel_file)

    # Padroniza coluna h
    if "h(cm)" in df.columns:
        df.rename(columns={"h(cm)": "h"}, inplace=True)

    # Decide o fluxo
    if 'h' in df.columns and 'theta' in df.columns:
        return code1(df, download_dir, plots_dir)
    else:
        return code2(df, download_dir, plots_dir)


def code1(df, download_dir, plots_dir):
    df['h'] = pd.to_numeric(df['h'], errors='coerce')
    df = df.dropna(subset=['h'])

    # Criação da coluna codeorign com todos os códigos
    df['codeorign'] = df['code']

    # Criar um DataFrame para solos descartados antes do sort
    soils_discarded = pd.DataFrame(columns=['codeorign', 'codedis'])

    codes_before_filtering = df['code'].unique()  # Captura todos os códigos antes da filtragem

    codes_with_h_0_1 = df.groupby('code').filter(lambda x: ((x['h'] >= 0) & (x['h'] <= 1)).any())['code'].unique()
    for code in codes_with_h_0_1:
        closest_theta_value = df[(df['code'] == code) & (df['h'] >= 0) & (df['h'] <= 1)].sort_values(by='h').iloc[0]['theta']
        df = pd.concat([df, pd.DataFrame({'code': [code], 'h': [0], 'theta': [closest_theta_value], 'codeorign': [code]})])

    df_head_0 = df[df['h'] == 0].groupby('code').head(1)

    df_head_30_80 = df[(df['h'] >= 30) & (df['h'] <= 80)].copy()
    df_head_30_80.loc[:, 'h_diff'] = abs(df_head_30_80['h'] - 60)
    df_head_30_80 = df_head_30_80.groupby('code').apply(lambda x: x.loc[x['h_diff'].idxmin()]).reset_index(drop=True)

    df_head_200_500 = df[(df['h'] >= 250) & (df['h'] <= 500)].copy()
    df_head_200_500.loc[:, 'h_diff'] = abs(df_head_200_500['h'] - 330)
    df_head_200_500 = df_head_200_500.groupby('code').apply(lambda x: x.loc[x['h_diff'].idxmin()]).reset_index(drop=True)

    df_head_9000_18000 = df[(df['h'] >= 9000) & (df['h'] <= 18000)].copy()
    df_head_9000_18000.loc[:, 'h_diff'] = abs(df_head_9000_18000['h'] - 15000)
    df_head_9000_18000 = df_head_9000_18000.groupby('code').apply(lambda x: x.loc[x['h_diff'].idxmin()]).reset_index(drop=True)

    df_sorted = pd.concat([df_head_0, df_head_30_80, df_head_200_500, df_head_9000_18000])
    grouped = df_sorted.groupby(['code', 'h'])['theta'].first().reset_index()

    grouped['PT'] = ''
    grouped.loc[grouped['h'] == 0, 'PT'] = grouped.loc[grouped['h'] == 0, 'theta']

    # Filtra códigos com pelo menos 4 parâmetros
    sample_ids_to_keep = grouped.groupby('code')['h'].count().loc[lambda x: x >= 4].index
    discarded_codes = set(codes_before_filtering) - set(sample_ids_to_keep)

    # Preencher o DataFrame de solos descartados
    soils_discarded['codeorign'] = list(codes_before_filtering)
    soils_discarded['codedis'] = [code if code in discarded_codes else None for code in codes_before_filtering]

    # Renomear a coluna codeorign para Sample_ID e garantir que os valores sejam texto
    soils_discarded['Sample_ID'] = soils_discarded['codeorign'].astype(str)
    soils_discarded.drop(columns=['codeorign'], inplace=True)

    # Criar a coluna OBS
    soils_discarded['OBS'] = soils_discarded['codedis'].apply(
        lambda x: "this soil could not be classified because it didn't meet requirements, see about us"
        if pd.notna(x) else '-'
    )

    grouped = grouped[grouped['code'].isin(sample_ids_to_keep)]

    sample_id_map = {sample_id: i + 1 for i, sample_id in enumerate(grouped['code'].unique())}
    grouped['code'] = grouped['code'].map(sample_id_map)

    # SoilSorted (DF)
    soil_sorted_df = grouped[['code', 'h', 'theta', 'PT']].copy()

    # SoilComplete (DF) com valores de h >= 30
    df_complete = df[df['code'].isin(sample_ids_to_keep) & (df['h'] >= 30)][['code', 'h', 'theta']].copy()
    df_complete['Sample_ID'] = df_complete['code']
    df_complete['code'] = df_complete['code'].map(sample_id_map)
    soil_complete_df = df_complete.copy()

    # SoilFull (DF) com todos os valores h >= 0
    df_full = df[df['code'].isin(sample_ids_to_keep)][['code', 'h', 'theta']].copy()
    df_full['Sample_ID'] = df_full['code']
    df_full['code'] = df_full['code'].map(sample_id_map)
    soil_full_df = df_full.copy()

    # Bounds/Params + saída final
    bounds_df, initialparam_df, ret_array = calculate_bounds(download_dir, soil_sorted_df)
    your_soil_classified_df = calculate_parameters(
        download_dir=download_dir,
        plots_dir=plots_dir,
        soil_sorted_df=soil_sorted_df,
        soil_complete_df=soil_complete_df,
        soil_full_df=soil_full_df,
        soils_discarded_df=soils_discarded.copy(),
        bounds_df=bounds_df,
        initialparam_df=initialparam_df,
        ret_array=ret_array
    )

    # Salvar APENAS o Excel final
    save_final_excel_only(download_dir, your_soil_classified_df)

    return your_soil_classified_df


def code2(df, download_dir, plots_dir):
    codes = []
    h_values = []
    theta_values = []

    # Criação da coluna codeorign com todos os códigos
    df['codeorign'] = df['code']  # Captura os códigos originais

    codes_before_filtering = df['code'].unique()  # Captura todos os códigos antes da filtragem

    # Criar um DataFrame para solos descartados antes do sort
    soils_discarded = pd.DataFrame(columns=['codeorign', 'codedis'])

    for index, row in df.iterrows():
        code = row['code']
        for column, value in row.items():
            if column != 'code' and not pd.isna(value):
                codes.append(code)
                h_values.append(column)
                theta_values.append(value)

    new_df = pd.DataFrame({'code': codes, 'h': h_values, 'theta': theta_values})
    new_df['h'] = pd.to_numeric(new_df['h'], errors='coerce')
    new_df = new_df.dropna(subset=['h'])

    # Criação da coluna codeorign no new_df
    new_df['codeorign'] = new_df['code']

    # Identificar os códigos que já possuem uma linha com h == 0
    codes_with_h_0 = new_df[new_df['h'] == 0]['code'].unique()

    codes_with_h_0_1 = new_df.groupby('code').filter(lambda x: ((x['h'] >= 0) & (x['h'] <= 1)).any())['code'].unique()
    for code in codes_with_h_0_1:
        if code not in codes_with_h_0:
            closest_theta_value = new_df[(new_df['code'] == code) & (new_df['h'] >= 0) & (new_df['h'] <= 1)].sort_values(by='h').iloc[0]['theta']
            new_df = pd.concat([new_df, pd.DataFrame({'code': [code], 'h': [0], 'theta': [closest_theta_value], 'codeorign': [code]})])

    df_head_0 = new_df[new_df['h'] == 0].groupby('code').head(1)

    df_head_30_80 = new_df[(new_df['h'] >= 30) & (new_df['h'] <= 80)].copy()
    df_head_30_80.loc[:, 'h_diff'] = abs(df_head_30_80['h'] - 60)
    df_head_30_80 = df_head_30_80.groupby('code').apply(lambda x: x.loc[x['h_diff'].idxmin()]).reset_index(drop=True)

    df_head_200_500 = new_df[(new_df['h'] >= 250) & (new_df['h'] <= 500)].copy()
    df_head_200_500.loc[:, 'h_diff'] = abs(df_head_200_500['h'] - 330)
    df_head_200_500 = df_head_200_500.groupby('code').apply(lambda x: x.loc[x['h_diff'].idxmin()]).reset_index(drop=True)

    df_head_9000_18000 = new_df[(new_df['h'] >= 9000) & (new_df['h'] <= 18000)].copy()
    df_head_9000_18000.loc[:, 'h_diff'] = abs(df_head_9000_18000['h'] - 15000)
    df_head_9000_18000 = df_head_9000_18000.groupby('code').apply(lambda x: x.loc[x['h_diff'].idxmin()]).reset_index(drop=True)

    # Concatenar os DataFrames e criar o final
    df_sorted = pd.concat([df_head_0, df_head_30_80, df_head_200_500, df_head_9000_18000])
    grouped = df_sorted.groupby(['code', 'h'])['theta'].first().reset_index()

    grouped['PT'] = ''
    grouped.loc[grouped['h'] == 0, 'PT'] = grouped.loc[grouped['h'] == 0, 'theta']

    # Filtra códigos com pelo menos 4 parâmetros
    sample_ids_to_keep = grouped.groupby('code')['h'].count().loc[lambda x: x >= 4].index
    discarded_codes = set(codes_before_filtering) - set(sample_ids_to_keep)

    # Preencher o DataFrame de solos descartados
    soils_discarded['codeorign'] = list(codes_before_filtering)
    soils_discarded['codedis'] = [code if code in discarded_codes else None for code in codes_before_filtering]

    # Renomear a coluna codeorign para Sample_ID e garantir que os valores sejam texto
    soils_discarded['Sample_ID'] = soils_discarded['codeorign'].astype(str)
    soils_discarded.drop(columns=['codeorign'], inplace=True)

    # Criar a coluna OBS
    soils_discarded['OBS'] = soils_discarded['codedis'].apply(
        lambda x: "this soil could not be classified because it didn't meet requirements, see about us"
        if pd.notna(x) else '-'
    )

    grouped = grouped[grouped['code'].isin(sample_ids_to_keep)]
    sample_id_map = {sample_id: i + 1 for i, sample_id in enumerate(grouped['code'].unique())}
    grouped['code'] = grouped['code'].map(sample_id_map)

    # SoilSorted (DF)
    soil_sorted_df = grouped[['code', 'h', 'theta', 'PT']].copy()

    # SoilComplete (DF) (no code2 original não filtrava h >= 30)
    df_complete = new_df[new_df['code'].isin(sample_ids_to_keep)][['code', 'h', 'theta']].copy()
    df_complete['Sample_ID'] = df_complete['code']
    df_complete['code'] = df_complete['code'].map(sample_id_map)
    soil_complete_df = df_complete.copy()

    # SoilFull (DF) com reenumeração
    df_full = new_df[new_df['code'].isin(sample_ids_to_keep)].copy()
    df_full[['code', 'h', 'theta', 'codeorign']] = df_full[['code', 'h', 'theta', 'codeorign']]
    sample_id_map_full = {sample_id: i + 1 for i, sample_id in enumerate(df_full['code'].unique())}
    df_full['code'] = df_full['code'].map(sample_id_map_full)
    # Mantém estrutura esperada adiante (inclui Sample_ID como no code1)
    df_full['Sample_ID'] = df_full['codeorign']
    soil_full_df = df_full[['code', 'h', 'theta', 'Sample_ID']].copy()

    bounds_df, initialparam_df, ret_array = calculate_bounds(download_dir, soil_sorted_df)
    your_soil_classified_df = calculate_parameters(
        download_dir=download_dir,
        plots_dir=plots_dir,
        soil_sorted_df=soil_sorted_df,
        soil_complete_df=soil_complete_df,
        soil_full_df=soil_full_df[['code', 'h', 'theta', 'Sample_ID']].copy(),
        soils_discarded_df=soils_discarded.copy(),
        bounds_df=bounds_df,
        initialparam_df=initialparam_df,
        ret_array=ret_array
    )

    # Salvar APENAS o Excel final
    save_final_excel_only(download_dir, your_soil_classified_df)

    return your_soil_classified_df


def calculate_bounds(download_dir, soil_sorted_df):
    df = soil_sorted_df.copy()

    # Calculate the alpha and T15000 values
    codes = []
    t15000_values = []
    alpha_values = []
    for code in df['code'].unique():
        df_temp = df[df['code'] == code].reset_index(drop=True)
        third_theta = df_temp.iloc[2]['theta']
        fourth_theta = df_temp.iloc[3]['theta']
        first_theta = df_temp.iloc[0]['theta']
        h_third_line = df_temp.iloc[2]['h']
        alpha = (((third_theta - (fourth_theta - 0.01)) / (first_theta - (fourth_theta - 0.01))) ** (-1 / 0.5) - 1) ** (1 - 0.5) / h_third_line
        t15000_temp = df_temp.iloc[3]['theta'] - 0.01
        codes.append(code)
        t15000_values.append(t15000_temp)
        alpha_values.append(alpha)

    initialparam_df = pd.DataFrame({'code': codes, 'T15000': t15000_values, 'alpha': alpha_values})

    # Mantém o TXT como no original
    initialparam_df.to_csv(os.path.join(download_dir, 'InitialParam.txt'), header=False, index=False, sep='\t')

    # Calculate bounds
    med0 = df[df['h'] == 0]['theta'].mean()
    med15000 = df[df['h'] >= 9000]['theta'].mean() - 0.01
    min0 = df[df['h'] == 0]['theta'].min()
    max0 = df[df['h'] == 0]['theta'].max()
    max15000 = df[df['h'] >= 9000]['theta'].max()

    bounds_df = pd.DataFrame({
        'Med0': [med0],
        'Med15000': [med15000],
        'Min0': [min0],
        'Max0': [max0],
        'Max15000': [max15000]
    })

    # (Não salva Bounds.xlsx: fica só em dataframe)

    # Transposição para ret.txt (mantém como no original, sem depender de Excel)
    transposed_data = []
    for site_id, group in df.groupby("code"):
        group = group.reset_index(drop=True)
        head = group["h"].head(16).reset_index(drop=True)
        matric_potential_values = list(head.combine_first(pd.Series([0] * 16)))

        theta = group["theta"].head(16).reset_index(drop=True)
        water_content_values = list(theta.combine_first(pd.Series([0] * 16)))

        transposed_data.append([site_id, len(group), *matric_potential_values, *water_content_values, group["PT"].iloc[0]])

    ret_path = os.path.join(download_dir, "ret.txt")
    with open(ret_path, "w") as f:
        for line in transposed_data:
            formatted_line = "\t".join(map(lambda x: '{:.3f}'.format(x) if isinstance(x, float) else str(x), line))
            f.write(formatted_line + "\n")

    ret_array = np.loadtxt(ret_path)

    return bounds_df, initialparam_df, ret_array


def calculate_parameters(
    download_dir,
    plots_dir,
    soil_sorted_df,
    soil_complete_df,
    soil_full_df,
    soils_discarded_df,
    bounds_df,
    initialparam_df,
    ret_array
):
    def vg4(par, h, par1):
        return np.abs(par[0]) + (par1 - np.abs(par[0])) / (1 + (par[1] * h) ** par[2]) ** (1 - 1 / par[2])

    def vg4ssq(par, press, theta, par1):
        theta1 = vg4(par, press, par1)
        diff = theta - theta1
        ssq = np.dot(diff, diff)
        return ssq

    def vg4ssq2(press, par0, par2, par3, par1):
        par = [par0, par2, par3]
        return vg4(par, press, par1)

    med15000_value = bounds_df['Med15000'].iloc[0]
    med0_value = bounds_df['Med0'].iloc[0]
    min0_value = bounds_df['Min0'].iloc[0]
    max15000_value = bounds_df['Max15000'].iloc[0]
    max0_value = bounds_df['Max0'].iloc[0]

    # df_iniciais agora vem em memória; ainda mantém o mesmo conteúdo do TXT
    df_iniciais = initialparam_df.copy()

    ret = ret_array
    MAXSAMP = ret.shape[0]
    MAXPNT = 3
    res2 = np.zeros((MAXSAMP, 4))
    rmse2 = np.zeros(MAXSAMP)
    nret = np.zeros(MAXSAMP)

    for i in range(MAXSAMP):
        id, npoints = int(ret[i, 0]), int(ret[i, 1])
        nret[i] = MAXPNT
        if npoints >= 4:
            theta = ret[i, 19:22]
            press = ret[i, 3:6]
            t15000_value = df_iniciais.loc[df_iniciais['code'] == id, 'T15000'].values[0]
            alpha_value = df_iniciais.loc[df_iniciais['code'] == id, 'alpha'].values[0]
            t15000_value = np.clip(t15000_value, 0, max15000_value)
            alpha_value = np.clip(alpha_value, 0.001, 10000)
            par1_value = ret[i, 18]
            par = np.array([t15000_value, alpha_value, 2])
            bounds = ([0, 0.001, 1], [max15000_value, 10000, 10])

            res2a, _ = curve_fit(
                lambda press, par0, par2, par3: vg4ssq2(press, par0, par2, par3, par1_value),
                press, theta, p0=par, bounds=bounds, method='trf', maxfev=100000, absolute_sigma=True
            )

            ssq2a = vg4ssq(res2a, press, theta, par1_value)
            res2[i, 0] = res2a[0]
            res2[i, 1] = par1_value
            res2[i, 2] = res2a[1]
            res2[i, 3] = res2a[2]
            rmse2[i] = np.sqrt(ssq2a / nret[i])
        else:
            res2[i, :] = -9.9
            rmse2[i] = -9.9

    # Mantém res2.txt como no original
    with open(os.path.join(download_dir, 'res2.txt'), 'wt+') as fres2:
        for i in range(MAXSAMP):
            fres2.write(f'{i + 1} {res2[i, 0]:.5f} {res2[i, 1]:.5f} {res2[i, 2]:.5f} {res2[i, 3]:.5f} {rmse2[i]:.5f}\n')

    columns = ['ThetaR', 'ThetaS', 'alpha', 'n']
    df_parameters = pd.DataFrame(res2, columns=columns)
    df_parameters['m'] = 1 - 1 / df_parameters['n']
    df_parameters['RMSE'] = rmse2
    df_parameters.insert(0, 'code', range(1, len(df_parameters) + 1))

    # ---------- SoilSorted_Head (DF) ----------
    df_ss = soil_sorted_df.copy().drop(columns=['PT'])

    def transpose_h_theta(df_):
        new_df = pd.DataFrame()
        for code in df_['code'].unique():
            code_data = df_[df_['code'] == code].reset_index(drop=True)
            h_values = code_data['h']
            theta_values = code_data['theta']
            for i, (h, theta) in enumerate(zip(h_values, theta_values)):
                new_df.at[code, f'h{i}'] = h
                new_df.at[code, f'theta{i}'] = theta
                new_df.at[code, 'code'] = code
        return new_df

    df_head = transpose_h_theta(df_ss)
    df_head = df_head[['code', 'h0', 'theta0', 'h1', 'theta1', 'h2', 'theta2', 'h3', 'theta3']].reset_index(drop=True)

    # ---------- RMSE total (DF) ----------
    df_sc = soil_complete_df.copy()
    df_merged = pd.merge(df_sc, df_parameters, on='code')

    def calculate_theta_calculated(row):
        h = row['h']
        thetaR = row['ThetaR']
        thetaS = row['ThetaS']
        alpha = row['alpha']
        m = row['m']
        n = row['n']
        return thetaR + (thetaS - thetaR) / (1 + (alpha * h) ** n) ** (m)

    df_merged['theta_calculated'] = df_merged.apply(calculate_theta_calculated, axis=1)

    def calculate_RMSETOTAL(row):
        code = row['code']
        subset = df_merged[df_merged['code'] == code]
        num_lines = len(subset)
        divisor = (num_lines - 3) if num_lines > 3 else 3
        sum_squared_diff = np.sum((subset['theta'] - subset['theta_calculated']) ** 2)
        return np.sqrt(sum_squared_diff / divisor)

    df_merged['TOTALRMSE'] = df_merged.apply(calculate_RMSETOTAL, axis=1)
    df_rmse = df_merged.copy()

    # ---------- ERRORMAX (DF) ----------
    df_soil = soil_sorted_df.copy()
    df_soil = df_soil[df_soil['h'] != 0]
    df_merged2 = pd.merge(df_soil, df_parameters, on='code', how='inner')
    df_merged2['theta_calculated'] = df_merged2['ThetaR'] + (df_merged2['ThetaS'] - df_merged2['ThetaR']) / (1 + (df_merged2['alpha'] * df_merged2['h']) ** df_merged2['n']) ** (df_merged2['m'])
    df_merged2['difference'] = abs(df_merged2['theta'] - df_merged2['theta_calculated'])
    df_errormax = df_merged2.groupby('code')['difference'].max().reset_index()
    df_errormax.rename(columns={'difference': 'ERRORMAX'}, inplace=True)
    df_errormax_full = df_errormax.copy()

    # ---------- Adherence (DF) ----------
    dfA = df_rmse.copy()

    first_errormax = df_errormax_full.groupby('code')['ERRORMAX'].first().reset_index()
    first_errormax.rename(columns={'ERRORMAX': 'First_ERRORMAX'}, inplace=True)
    dfA = pd.merge(dfA, first_errormax, on='code')
    dfA['ERRORMAX'] = dfA['First_ERRORMAX']
    dfA.drop(columns=['First_ERRORMAX'], inplace=True)

    dfA['W60'] = ((1 + (dfA['alpha'] * 60) ** dfA['n']) ** -dfA['m'])
    dfA['W15000'] = ((1 + (dfA['alpha'] * 15000) ** dfA['n']) ** -dfA['m'])
    dfA['A60'] = 1 - dfA['W60']
    dfA['W60-W15000'] = dfA['W60'] - dfA['W15000']
    dfA['W60%'] = dfA['W60'] * 100
    dfA['W15000%'] = dfA['W15000'] * 100
    dfA['A60%'] = 100 - dfA['W60%']
    dfA['W60%-W15000%'] = dfA['W60%'] - dfA['W15000%']

    dfA.drop(columns=['h', 'theta', 'theta_calculated'], inplace=True)
    dfA = dfA.groupby('code').first().reset_index()

    dfA['Adherence'] = np.where(
        (dfA['ERRORMAX'] <= 0.02) & (dfA['TOTALRMSE'] <= 0.035), 'High',
        np.where((dfA['ERRORMAX'] > 0.04) | (dfA['TOTALRMSE'] > 0.07), 'Low', 'Medium')
    )
    dfA['Classification'] = np.where(
        dfA['Adherence'] == 'High', 'Genuine Soil',
        np.where(dfA['Adherence'] == 'Medium', 'Adopted Soil', 'Rejected Soil')
    )

    dfA = pd.merge(df_head, dfA, on='code', how='inner')

    # ---------- Classificação adicional ----------
    df = dfA.copy()

    def classify_suborder(row):
        diff = row['theta0'] - row['ThetaR']
        if 0 < diff <= 0.20:
            return 1
        elif 0.20 < diff <= 0.40:
            return 2
        elif 0.40 < diff <= 0.60:
            return 3
        elif diff > 0.60:
            return 4
        else:
            return None

    def classify_order(row):
        if 2/3 <= row["A60"] < 1 and 0 < row['W60-W15000'] < 1/3 and 0 < row['W15000'] < 1/3:
            return "A"
        elif 1/3 <= row["A60"] < 2/3 and 0 < row['W60-W15000'] < 1/3 and 0 < row['W15000'] < 1/3:
            return "D"
        elif 1/3 <= row["A60"] < 2/3 and 1/3 <= row['W60-W15000'] < 2/3 and 0 < row['W15000'] < 1/3:
            return "B"
        elif 0 < row["A60"] < 1/3 and 2/3 <= row['W60-W15000'] < 1 and 0 < row['W15000'] < 1/3:
            return "C"
        elif 0 < row["A60"] < 1/3 and 1/3 <= row['W60-W15000'] < 2/3 and 0 < row['W15000'] < 1/3:
            return "E"
        elif 0 < row["A60"] < 1/3 and 1/3 <= row['W60-W15000'] < 2/3 and 1/3 <= row['W15000'] < 2/3:
            return "G"
        elif 0 < row["A60"] < 1/3 and 0 < row['W60-W15000'] < 1/3 and 2/3 <= row['W15000'] < 1:
            return "I"
        elif 0 < row["A60"] < 1/3 and 0 < row['W60-W15000'] < 1/3 and 1/3 <= row['W15000'] < 2/3:
            return "H"
        elif 1/3 <= row["A60"] < 2/3 and 0 < row['W60-W15000'] < 1/3 and 1/3 <= row['W15000'] < 2/3:
            return "F"
        else:
            return "Error"

    df["Order"] = df.apply(classify_order, axis=1)
    df["Suborder"] = df.apply(classify_suborder, axis=1)
    df["Suborder"] = df["Suborder"].fillna(0).astype(int)
    df["Family"] = df["Order"] + df["Suborder"].astype(str)

    suborder_dict = {
        1: "Low Effective Porosity",
        2: "Moderate Effective Porosity",
        3: "High Effective Porosity",
        4: "Very High Effective Porosity"
    }
    order_dict = {
        "A": "Highly Macrospacious Soil",
        "C": "Highly Mesospacious Soil",
        "I": "Highly Microspacious Soil",
        "D": "Macrospacious Soil",
        "E": "Mesospacious Soil",
        "H": "Microspacious Soil",
        "B": "Macro-Mesospacious Soil",
        "F": "Macro-Microspacious Soil",
        "G": "Meso-Microspacious Soil"
    }

    def get_family_nomenclature(row):
        porosity = suborder_dict.get(row['Suborder'], "Unknown Porosity")
        order_name = order_dict.get(row['Order'], "Unknown Soil Type")
        return f"{porosity} - {order_name}"

    df["Family Nomenclature"] = df.apply(get_family_nomenclature, axis=1)

    df['a (cm³/cm³)'] = df['theta0'] - df['theta2']
    df['w (cm³/cm³)'] = df['theta2'] - df['theta3']
    df['Ksat (cm/d)'] = 1931 * (df['a (cm³/cm³)'] ** 1.948)

    def classify_a_value(a):
        if a < 0.10:
            return "Low"
        elif 0.10 <= a <= 0.20:
            return "Moderate"
        elif a > 0.20:
            return "High"
        else:
            return "Undefined"

    df['a value'] = df['a (cm³/cm³)'].apply(classify_a_value)

    def classify_w_value(w):
        if w < 0.06:
            return "Low"
        elif 0.06 <= w <= 0.12:
            return "Moderate"
        elif w > 0.12:
            return "High"
        else:
            return "Undefined"

    df['w value'] = df['w (cm³/cm³)'].apply(classify_w_value)

    def classify_ksat_value(ksat):
        if ksat <= 12:
            return "Slow"
        elif 12 < ksat <= 120:
            return "Moderate"
        elif ksat > 120:
            return "Rapid"
        else:
            return "Undefined"

    df['ksat value'] = df['Ksat (cm/d)'].apply(classify_ksat_value)

    def classify_hydraulic_class(row):
        a_val = row['a value']
        w_val = row['w value']
        ksat_val = row['ksat value']

        if (a_val == "Low" and w_val in ["Low", "Moderate", "High"]):
            restriction = "Soil with air restriction"
        elif (a_val in ["High", "Moderate"] and w_val == "Low"):
            restriction = "Soil with water restriction"
        elif ((a_val == "Moderate" and w_val == "High") or
              (a_val == "High" and w_val in ["Moderate", "High"]) or
              (a_val == "Moderate" and w_val == "Moderate")):
            restriction = "Soil without air and water restriction"
        else:
            restriction = "Undefined Restriction"

        if ksat_val == "Slow":
            permeability = "slow permeability"
        elif ksat_val == "Moderate":
            permeability = "moderate permeability"
        elif ksat_val == "Rapid":
            permeability = "rapid permeability"
        else:
            permeability = "undefined permeability"

        return f"{restriction} and {permeability}"

    df['Hydraulic Class'] = df.apply(classify_hydraulic_class, axis=1)

    df.drop(columns=['a value', 'w value', 'ksat value', 'RMSE'], inplace=True)

    # Rename the columns as requested
    df.rename(columns={"TOTALRMSE": "RMSE_30cm-15000cm"}, inplace=True)

    # Reorder the columns to have 'Sample_ID' as the first column
    columns_order = ['Sample_ID'] + [col for col in df.columns if col != 'Sample_ID']
    df = df[columns_order]

    # Convert the column SAMPLE_ID to text
    df['Sample_ID'] = df['Sample_ID'].astype(str)

    # ---------- PLOTS (NÃO ALTERADO) ----------
    soilfull_df = soil_full_df.copy()
    adherence_df = dfA.copy()  # usa o DF equivalente ao Adherence.xlsx

    gc.collect()

    soilfull_df['log_h'] = np.log10(soilfull_df['h'].replace(0, np.nan))
    soilfull_df['Theta'] = soilfull_df['theta']

    unique_codes = soilfull_df['code'].unique()
    pdf_path = os.path.join(plots_dir, 'retentioncurveplots.pdf')

    os.makedirs(plots_dir, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        num_plots_per_page = 9
        plots_per_row = 3
        rows = num_plots_per_page // plots_per_row

        for i, code in enumerate(unique_codes):
            if i % num_plots_per_page == 0:
                fig, axes = plt.subplots(rows, plots_per_row, figsize=(15, 15))
                axes = axes.flatten()

            code_df = soilfull_df[soilfull_df['code'] == code]
            sample_id = adherence_df[adherence_df['code'] == code]['Sample_ID'].values[0]

            ax = axes[i % num_plots_per_page]
            plot_color = 'black'

            for index, row in code_df.iterrows():
                if row['h'] == 0:
                    ax.scatter(0, row['Theta'], color=plot_color, marker='x', s=100, label='φ', clip_on=False)
                else:
                    ax.scatter(row['log_h'], row['Theta'], color=plot_color, alpha=0.6)

            adherence_row = adherence_df[adherence_df['code'] == code]
            if not adherence_row.empty:
                ThetaR = adherence_row['ThetaR'].values[0]
                ThetaS = adherence_row['ThetaS'].values[0]
                alpha = adherence_row['alpha'].values[0]
                n = adherence_row['n'].values[0]

                h_values_above_30 = np.linspace(30, soilfull_df['h'].max(), 100)
                Theta_values_above_30 = ThetaR + (ThetaS - ThetaR) / (1 + (alpha * h_values_above_30) ** n) ** (1 - 1 / n)

                ax.scatter(np.log10(adherence_row['h1'].values[0]), adherence_row['theta1'].values[0], color='black', marker='s', s=100)
                ax.scatter(np.log10(adherence_row['h2'].values[0]), adherence_row['theta2'].values[0], color='black', marker='s', s=100)
                ax.scatter(np.log10(adherence_row['h3'].values[0]), adherence_row['theta3'].values[0], color='black', marker='s', s=100)

                ax.plot(np.log10(h_values_above_30), Theta_values_above_30, linestyle='--', color='gray')

                error_max = round(adherence_row['ERRORMAX'].values[0], 3)
                total_rmse = round(adherence_row['TOTALRMSE'].values[0], 3)

                handles = [
                    plt.Line2D([0], [0], color='w', label=f'ERRORMAX: {error_max} cm³/cm³'),
                    plt.Line2D([0], [0], color='w', label=f'RRMSE$_{{30-18000}}$: {total_rmse} cm³/cm³'),
                    plt.Line2D([0], [0], marker='s', color='black', label='θ1, θ2, θ3', markerfacecolor='black', markersize=10, linestyle='None'),
                    plt.Line2D([0], [0], marker='x', color='black', label='φ', markersize=10, linestyle='None'),
                    plt.Line2D([0], [0], marker='o', color='gray', label='θ', markersize=6, linestyle='None')
                ]
                ax.legend(handles=handles, loc='upper right')

            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel("log h (cm)")
            ax.set_ylabel("Theta (cm³/cm³)")
            ax.set_title(f"Retention curve - Soil {sample_id}")
            ax.grid()
            ax.set_xlim(left=0)
            ax.set_ylim(0, 1.0)

            if (i + 1) % num_plots_per_page == 0 or i == len(unique_codes) - 1:
                for j in range((i % num_plots_per_page) + 1, num_plots_per_page):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    def map_color(adherence):
        if adherence == 'High':
            return 'green'
        elif adherence == 'Medium':
            return '#FFC300'
        elif adherence == 'Low':
            return 'red'

    def plot_ternary(df_, title, filename):
        scale = 100
        try:
            figure, tax = ternary.figure(scale=scale)
        except AttributeError:
            raise Exception("A função 'figure' não está disponível no módulo 'ternary'. Verifique a instalação.")

        figure.set_size_inches(5, 5)

        step = scale / 9
        grid_color = '#a9a9a9'

        for i in range(9):
            for j in range(9 - i):
                x1, y1 = i * step, j * step
                x2, y2 = (i + 1) * step, j * step
                x3, y3 = i * step, (j + 1) * step

                tax.line((x1, y1), (x2, y2), color=grid_color, linestyle='--', linewidth=0.5)
                tax.line((x2, y2), (x3, y3), color=grid_color, linestyle='--', linewidth=0.5)
                tax.line((x3, y3), (x1, y1), color=grid_color, linestyle='--', linewidth=0.5)

        tax.boundary(linewidth=1.0)

        fontsize = 12
        offset = 0.2
        tax.left_axis_label("    W15000 (%)", fontsize=fontsize, offset=offset, rotation=-240)
        tax.right_axis_label("W60-W15000 (%)", fontsize=fontsize, offset=offset, rotation=240)
        tax.bottom_axis_label("      A60 (100-W60) (%)", fontsize=fontsize, offset=offset, rotation=180)

        tax.annotate("Microspace (%)", position=(62, 54), fontsize=10, rotation=60, va='center', ha='center')
        tax.annotate("Mesospace (%)", position=(-11.6, 54), fontsize=10, rotation=-60, va='center', ha='center')
        tax.annotate("Macrospace (%)", position=(54, -11.6), fontsize=10, rotation=0, va='center', ha='center')

        for _, row in df_.iterrows():
            color = map_color(row['Adherence'])
            tax.scatter([[row['A60%'], row['W15000%'], row['W60%-W15000%']]], marker='o', s=35, color=color)

        plt.gca().invert_xaxis()
        triangles = [((66.7, 0), (66.7, 33.3)), ((33.3, 0), (33.3, 66.7)), ((66.7, 33.3), (0, 33.3)),
                     ((33.3, 66.7), (0, 66.7)), ((0, 66.7), (66.7, 0)), ((0, 33.3), (33.3, 0))]
        for p1, p2 in triangles:
            tax.line(p1, p2, linewidth=1, color='black')

        titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        midpoints = [(81, 9, 5), (47, 9.5), (13, 9.5), (59, 20), (24, 20), (45, 45), (13, 45), (26, 52), (12, 78)]
        for title_point, point in zip(titles, midpoints):
            tax.annotate(title_point, point)

        plt.title(title, pad=20)

        tax.ticks(axis='lbr', ticks=[0, 33.3, 66.7, 100], linewidth=1, offset=0.025)
        tax.get_axes().axis('off')
        tax.clear_matplotlib_ticks()

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Genuine Soil', markerfacecolor='green', markersize=5),
            plt.Line2D([0], [0], marker='o', color='w', label='Adopted Soil', markerfacecolor='#FFC300', markersize=5),
            plt.Line2D([0], [0], marker='o', color='w', label='Rejected Soil', markerfacecolor='red', markersize=5)
        ]
        plt.legend(handles=legend_elements, loc='upper left', title="Adherence", fontsize='small')

        plt.savefig(filename, format='jpg')
        plt.close()

    plot_ternary(df, "ALL SUBORDERS", os.path.join(plots_dir, 'allsuborders_plot.jpg'))

    for order in range(1, 5):
        filtered_df = df[df.apply(lambda row: classify_suborder(row) == order, axis=1)]
        plot_ternary(filtered_df, f"Suborder {order}", os.path.join(plots_dir, f'suborder_{order}_plot.jpg'))

    files_to_include = [
        'allsuborders_plot.jpg',
        'suborder_1_plot.jpg',
        'suborder_2_plot.jpg',
        'suborder_3_plot.jpg',
        'suborder_4_plot.jpg',
        'retentioncurveplots.pdf'
    ]

    with zipfile.ZipFile(os.path.join(download_dir, 'ternary_plots.zip'), 'w') as zipf:
        for root, _, files in os.walk(plots_dir):
            for file in files:
                if file in files_to_include:
                    zipf.write(os.path.join(root, file), file)

    # ---------- Formatação/colunas finais (como no original) ----------
    df.rename(columns={"h0": "h0 (cm)",
                       "theta0": "theta0 (cm³/cm³)",
                       "h1": "h1 (cm)",
                       "theta1": "theta1 (cm³/cm³)",
                       "h2": "h2 (cm)",
                       "theta2": "theta2 (cm³/cm³)",
                       "h3": "h3 (cm)",
                       "theta3": "theta3 (cm³/cm³)",
                       "ThetaR": "ThetaR (cm³/cm³)",
                       "ThetaS": "ThetaS (cm³/cm³)",
                       "alpha": "alpha (cm-1)",
                       "W15000%": "Microspace%",
                       "A60%": "Macrospace%",
                       "W60%-W15000%": "Mesospace%",
                       "Classification": "Sample status",
                       "Family Nomenclature": "Family Name"}, inplace=True)

    df[["RMSE_30cm-15000cm", "ERRORMAX"]] = df[["RMSE_30cm-15000cm", "ERRORMAX"]].round(4)
    df[["Microspace%", "Macrospace%", "Mesospace%"]] = df[["Microspace%", "Macrospace%", "Mesospace%"]].round(1)

    df.drop(columns=['Adherence', 'Order', 'Suborder', 'W60', 'W15000', 'A60', 'W60-W15000', 'W60%'], inplace=True)

    # Junta com descartados (como no original)
    your_soil_classified = df.copy()
    soils_discarded = soils_discarded_df.copy()

    merged_df = pd.merge(your_soil_classified, soils_discarded[['Sample_ID', 'OBS']],
                         how='outer', on='Sample_ID')

    column_order = ['Sample_ID', 'OBS'] + [col for col in merged_df.columns if col not in ['Sample_ID', 'OBS']]
    merged_df = merged_df[column_order]
    your_soil_classified = merged_df

    columns_to_remove = ["code", "RMSE_3points", "Adherence", "W60", "W15000", "A60", "W60-W15000"]
    your_soil_classified = your_soil_classified.drop(columns=[col for col in columns_to_remove if col in your_soil_classified.columns])

    your_soil_classified = your_soil_classified.rename(columns={"Sample_ID": "code"})

    return your_soil_classified


def save_final_excel_only(download_dir, your_soil_classified_df):
    output_path = os.path.join(download_dir, 'YourSoilClassified.xlsx')

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        your_soil_classified_df.to_excel(writer, index=False)

        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column_letter].width = adjusted_width
            for cell in column:
                cell.alignment = cell.alignment.copy(horizontal='center', wrap_text=True)
