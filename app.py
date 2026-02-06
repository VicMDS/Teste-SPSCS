from flask import Flask, request, send_from_directory, jsonify, render_template, session, after_this_request
import os
import pandas as pd
import tempfile
import uuid
import time
import shutil

# Mantém ASSETS_FOLDER fixo (normalmente arquivos estáticos do app)
from spscs.config import ASSETS_FOLDER

# ==========================================================
# ADIÇÕES NECESSÁRIAS PARA COMPATIBILIZAR COM O CÓDIGO NOVO
# (sem mexer no resto do seu arquivo)
# - O código novo não salva SoilSorted/SoilComplete/SoilFull como Excel no pipeline,
#   mas o endpoint /create_excel ainda cria esses arquivos "mantido".
# - Criamos wrappers calculate_bounds2/calculate_parameters2 somente se não existirem,
#   usando as novas funções calculate_bounds/calculate_parameters (em memória),
#   e salvando apenas o YourSoilClassified.xlsx (como no código novo).
# ==========================================================

# Cache simples para ret_array gerado em calculate_bounds2 (por pasta do usuário)
_RET_CACHE = {}

try:
    # Se você já tiver calculate_bounds2/calculate_parameters2 no seu pacote, usa eles.
    from spscs.functions import calculate_bounds2, calculate_parameters2, process_excel
except Exception:
    # Caso contrário, usamos o "process_excel" novo + as funções novas (em memória)
    from spscs.functions import (
        process_excel,
        calculate_bounds,
        calculate_parameters,
        save_final_excel_only
    )

    def calculate_bounds2(download_dir):
        """
        Wrapper compatível com o fluxo antigo:
        - Lê SoilSorted.xlsx (que o endpoint /create_excel ainda cria)
        - Chama calculate_bounds(download_dir, soil_sorted_df) do código novo
        - Retorna (df_soil_sorted, df_bounds, df_iniciais)
        """
        soil_sorted_path = os.path.join(download_dir, "SoilSorted.xlsx")
        df_soil_sorted = pd.read_excel(soil_sorted_path)

        # calculate_bounds novo retorna: (bounds_df, initialparam_df, ret_array)
        df_bounds, df_iniciais, ret_array = calculate_bounds(download_dir, df_soil_sorted)

        # guarda ret_array para o próximo passo (calculate_parameters2)
        _RET_CACHE[download_dir] = ret_array

        return df_soil_sorted, df_bounds, df_iniciais

    def calculate_parameters2(download_dir, plots_dir, df_soil_sorted, df_bounds, df_iniciais):
        """
        Wrapper compatível com o fluxo antigo:
        - Lê SoilComplete.xlsx e SoilFull.xlsx (que o endpoint /create_excel ainda cria)
        - Cria um soils_discarded_df mínimo (sem descartes, OBS='-')
        - Recupera ret_array do cache
        - Chama calculate_parameters(...) do código novo
        - Salva APENAS YourSoilClassified.xlsx (como o código novo)
        - NÃO altera nada dos plots (a função nova gera os plots/zip igual)
        """
        soil_complete_path = os.path.join(download_dir, "SoilComplete.xlsx")
        soil_full_path = os.path.join(download_dir, "SoilFull.xlsx")

        soil_complete_df = pd.read_excel(soil_complete_path)
        soil_full_df = pd.read_excel(soil_full_path)

        # Garante Sample_ID como no pipeline novo (usado adiante em colunas/merge)
        if "Sample_ID" not in soil_complete_df.columns:
            if "SampleID" in soil_complete_df.columns:
                soil_complete_df["Sample_ID"] = soil_complete_df["SampleID"]
            else:
                soil_complete_df["Sample_ID"] = soil_complete_df["code"]

        if "Sample_ID" not in soil_full_df.columns:
            if "SampleID" in soil_full_df.columns:
                soil_full_df["Sample_ID"] = soil_full_df["SampleID"]
            else:
                soil_full_df["Sample_ID"] = soil_full_df["code"]

        soils_discarded_df = pd.DataFrame({
            "Sample_ID": soil_full_df["Sample_ID"].astype(str).unique(),
            "codedis": [None] * len(soil_full_df["Sample_ID"].astype(str).unique()),
        })
        soils_discarded_df["OBS"] = "-"

        ret_array = _RET_CACHE.get(download_dir)
        if ret_array is None:
            # Se por algum motivo não existir, recria via calculate_bounds2
            _, _, _ = calculate_bounds2(download_dir)
            ret_array = _RET_CACHE.get(download_dir)

        your_soil_classified_df = calculate_parameters(
            download_dir=download_dir,
            plots_dir=plots_dir,
            soil_sorted_df=df_soil_sorted,
            soil_complete_df=soil_complete_df,
            soil_full_df=soil_full_df[["code", "h", "theta", "Sample_ID"]].copy(),
            soils_discarded_df=soils_discarded_df.copy(),
            bounds_df=df_bounds,
            initialparam_df=df_iniciais,
            ret_array=ret_array
        )

        # Salvar APENAS o Excel final
        save_final_excel_only(download_dir, your_soil_classified_df)

        return your_soil_classified_df

# ==========================================================
# FIM DAS ADIÇÕES
# ==========================================================


app = Flask(__name__)

# Necessário para usar session (cookie)
# Em produção, use uma chave segura via variável de ambiente.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


def _ensure_user_id() -> str:
    """Garante um identificador único por usuário (sessão)."""
    if "user_uuid" not in session:
        session["user_uuid"] = str(uuid.uuid4())
    return session["user_uuid"]


def _cleanup_old_temp_dirs(max_age_seconds: int = 2 * 60 * 60) -> None:
    """
    Remove diretórios temporários antigos (por idade) para evitar acúmulo.
    max_age_seconds: idade máxima em segundos (padrão: 2 horas).
    """
    root = os.path.join(tempfile.gettempdir(), "spscs")
    if not os.path.isdir(root):
        return

    now = time.time()
    try:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if (now - mtime) > max_age_seconds:
                shutil.rmtree(path, ignore_errors=True)
    except OSError:
        # Se houver qualquer problema de permissão ou concorrência, apenas ignora
        return


def _get_user_base_dir() -> str:
    """Base temporária exclusiva do usuário."""
    _cleanup_old_temp_dirs()
    user_id = _ensure_user_id()
    base = os.path.join(tempfile.gettempdir(), "spscs", user_id)
    os.makedirs(base, exist_ok=True)
    return base


def _get_user_dirs():
    """
    Mantém os nomes DOWNLOAD_FOLDER / PLOTS_FOLDER / UPLOAD_FOLDER,
    mas agora são diretórios temporários por usuário (sessão) e criados sob demanda.
    """
    base = _get_user_base_dir()

    DOWNLOAD_FOLDER = os.path.join(base, "download")
    PLOTS_FOLDER = os.path.join(base, "plots")
    UPLOAD_FOLDER = os.path.join(base, "upload")

    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    return DOWNLOAD_FOLDER, PLOTS_FOLDER, UPLOAD_FOLDER


def _delete_user_dir_after_response():
    user_id = session.get("user_uuid")
    if not user_id:
        return

    base = os.path.join(tempfile.gettempdir(), "spscs", user_id)

    @after_this_request
    def _remove_dir(response):
        shutil.rmtree(base, ignore_errors=True)
        return response


@app.route('/')
def index():
    _ensure_user_id()
    return render_template('index.html')


@app.route('/create_excel', methods=['POST'])
def create_excel():
    DOWNLOAD_FOLDER, PLOTS_FOLDER, UPLOAD_FOLDER = _get_user_dirs()

    data = request.json
    soil_code = data['soil_code']
    num_points = data['num_points']
    h_values = data['h_values']
    theta_values = data['theta_values']

    # Validação mínima para evitar inconsistências
    if not isinstance(h_values, list) or not isinstance(theta_values, list):
        return jsonify({'success': False, 'message': 'h_values and theta_values must be lists.'})
    if len(h_values) != num_points or len(theta_values) != num_points:
        return jsonify({'success': False, 'message': 'num_points does not match the length of values.'})

    # Criação do DataFrame
    df = pd.DataFrame({
        'SampleID': [soil_code] * num_points,
        'code': [1] * num_points,
        'h': h_values,
        'theta': theta_values
    })

    # Salvando o arquivo Excel (mantido)
    filepath = os.path.join(DOWNLOAD_FOLDER, 'YourSoil.xlsx')
    df.to_excel(filepath, index=False)

    # Criar SoilFull (mantido: usado nas plotagens)
    soil_full_path = os.path.join(DOWNLOAD_FOLDER, 'SoilFull.xlsx')
    df.to_excel(soil_full_path, index=False)

    # Criar SoilComplete (mantido: usado no cálculo de RMSE total)
    soil_complete = df[df['h'] >= 30]
    soil_complete_path = os.path.join(DOWNLOAD_FOLDER, 'SoilComplete.xlsx')
    soil_complete.to_excel(soil_complete_path, index=False)

    # Criar SoilSorted (mantido: funções esperam esse nome/arquivo como entrada)
    soil_sorted = pd.DataFrame(columns=['code', 'h', 'theta', 'PT'])

    # Primeiro ponto
    if ((df['h'] >= 0) & (df['h'] <= 1)).any():
        first_point = df[(df['h'] >= 0) & (df['h'] <= 1)]
        closest_to_zero = first_point.iloc[(first_point['h']).abs().argsort()[:1]][['code', 'h', 'theta']]
        soil_sorted = pd.concat([soil_sorted, closest_to_zero], ignore_index=True)
    else:
        return jsonify({'success': False, 'message': 'No points with h between 0 and 1.'})

    # Segundo ponto
    second_point = df[(df['h'] >= 30) & (df['h'] <= 80)]
    if not second_point.empty:
        second_point = second_point.iloc[(second_point['h'] - 60).abs().argsort()[:1]][['code', 'h', 'theta']]
        soil_sorted = pd.concat([soil_sorted, second_point], ignore_index=True)

    # Terceiro ponto
    third_point = df[(df['h'] >= 250) & (df['h'] <= 500)]
    if not third_point.empty:
        third_point = third_point.iloc[(third_point['h'] - 330).abs().argsort()[:1]][['code', 'h', 'theta']]
        soil_sorted = pd.concat([soil_sorted, third_point], ignore_index=True)

    # Quarto ponto
    fourth_point = df[(df['h'] >= 9000) & (df['h'] <= 18000)]
    if not fourth_point.empty:
        fourth_point = fourth_point.iloc[(fourth_point['h'] - 15000).abs().argsort()[:1]][['code', 'h', 'theta']]
        soil_sorted = pd.concat([soil_sorted, fourth_point], ignore_index=True)

    # Verificação de pontos encontrados
    if len(soil_sorted) < 4:
        return jsonify({'success': False, 'message': 'SoilSorted requires at least 4 points.'})

    # Criar a nova coluna PT (mantido)
    soil_sorted['PT'] = [soil_sorted['theta'].iloc[0]] + [''] * (len(soil_sorted) - 1)

    # Remover a coluna SampleID (mantido)
    soil_sorted = soil_sorted.drop(columns=['SampleID'], errors='ignore')

    soil_sorted_path = os.path.join(DOWNLOAD_FOLDER, 'SoilSorted.xlsx')
    soil_sorted.to_excel(soil_sorted_path, index=False)

    # -------------------------------------------------------
    # AQUI É A ÚNICA ALTERAÇÃO NECESSÁRIA PARA "CASAR" COM O CÓDIGO NOVO
    # -------------------------------------------------------
    # calculate_bounds2 agora RETORNA dataframes em memória
    df_soil_sorted, df_bounds, df_iniciais = calculate_bounds2(DOWNLOAD_FOLDER)

    # calculate_parameters2 agora RECEBE os dataframes e continua salvando:
    # - imagens em PLOTS_FOLDER
    # - zip em DOWNLOAD_FOLDER
    # - e SOMENTE o YourSoilClassified.xlsx no DOWNLOAD_FOLDER
    calculate_parameters2(
        DOWNLOAD_FOLDER,
        PLOTS_FOLDER,
        df_soil_sorted=df_soil_sorted,
        df_bounds=df_bounds,
        df_iniciais=df_iniciais
    )

    return jsonify({'success': True, 'message': 'Excel files created successfully.'})


@app.route('/upload', methods=['POST'])
def upload_file():
    DOWNLOAD_FOLDER, PLOTS_FOLDER, UPLOAD_FOLDER = _get_user_dirs()

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if file:
        unique_name = f"{uuid.uuid4()}__{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        # Mantido: process_excel (assumindo que já está compatível ou você também vai ajustar nele)
        process_excel(filepath, download_dir=DOWNLOAD_FOLDER, plots_dir=PLOTS_FOLDER)

        file1 = '/download_xlsx'
        file2 = '/download_zip'

        return jsonify({'success': True, 'file1': file1, 'file2': file2})


@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(ASSETS_FOLDER, filename)


@app.route('/download_xlsx')
def download_xlsx():
    DOWNLOAD_FOLDER, PLOTS_FOLDER, UPLOAD_FOLDER = _get_user_dirs()
    return send_from_directory(DOWNLOAD_FOLDER, 'YourSoilClassified.xlsx', as_attachment=True)


@app.route('/download_zip')
def download_zip():
    DOWNLOAD_FOLDER, PLOTS_FOLDER, UPLOAD_FOLDER = _get_user_dirs()
    _delete_user_dir_after_response()
    return send_from_directory(DOWNLOAD_FOLDER, 'ternary_plots.zip', as_attachment=True)


@app.route('/ternary_plots/<filename>')
def ternary_plots(filename):
    DOWNLOAD_FOLDER, PLOTS_FOLDER, UPLOAD_FOLDER = _get_user_dirs()
    return send_from_directory(PLOTS_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)



