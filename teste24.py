import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import requests

# URL base do repositório GitHub
BASE_URL = "https://raw.githubusercontent.com/ElmerDotti/HNK/main/"

# Função para carregar arquivos do GitHub
def carregar_arquivo(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

# Função de login com correção para carregamento da imagem
def login():
    try:
        response = requests.get(f"{BASE_URL}logo-removebg.png")
        response.raise_for_status()  # Verifica se a resposta está OK
        logo = Image.open(BytesIO(response.content))  # Carrega a imagem a partir dos bytes
        st.image(logo, width=150)  # Exibe o logotipo da Heineken
    except requests.exceptions.RequestException as e:
        st.error("Erro ao carregar o logotipo. Verifique a URL.")
        st.write(e)
    except Image.UnidentifiedImageError:
        st.error("Erro ao carregar a imagem: formato de imagem não identificado.")

    st.title("Login")

    # Campos de login
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")

    # Autenticação simples
    if st.button("Entrar"):
        if usuario == "HNK" and senha == "HNK123":
            st.session_state["login"] = True
            st.success("Login realizado com sucesso!")
        else:
            st.error("Usuário ou senha incorretos!")

# Configuração de estilo para exibir as etapas do método CRISP-DM em fonte branca
def etapa_crisp_dm(texto, descricao, objetivo_tecnica):
    st.markdown(f"<h5 style='color:white;'>{texto}</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:white;'>{descricao}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:white;'><strong>Objetivo:</strong> {objetivo_tecnica}</p>", unsafe_allow_html=True)

# Mapeamento de etapas de fabricação para rótulos de string
ETAPAS_NOMES = {
    0: "Moinho", 1: "Mistura", 2: "Cozimento", 3: "Filtragem",
    4: "Fervura", 5: "Clarificação", 6: "Resfriamento",
    7: "Fermentação", 8: "Maturação", 9: "Envase"
}

# Carregar e filtrar dados para o produto 'AMSTEL'
def carregar_dados():
    try:
        caminho = BASE_URL + "Heineken%20-%20Data%20Science%20CB%20Use%20Case%202024.csv"
        dados = pd.read_csv(caminho)
        dados = dados[dados['Product'] == 'AMSTL']
        dados['Date/Time'] = pd.to_datetime(dados['Date/Time'])
        return dados
    except Exception as e:
        st.error("Erro ao carregar os dados.")
        st.write(e)
        return pd.DataFrame()  # Retorna um DataFrame vazio se houver erro

# Tratamento de dados: interpolação e valores absolutos para negativos
def tratar_dados(dados):
    if dados.empty:
        st.warning("Os dados carregados estão vazios. Verifique o arquivo de dados.")
        return pd.DataFrame()
    
    dados = dados.copy()
    dados.interpolate(method='linear', inplace=True)
    dados.update(dados.select_dtypes(include=[np.number]).abs())
    return dados

# Segmentação das etapas de fabricação com KMeans
def segmentar_etapas(dados, n_clusters=10):
    # Seleciona apenas as colunas numéricas para a segmentação
    dados_numericos = dados.select_dtypes(include=[np.number])
    
    # Verificação adicional para garantir que o DataFrame numérico não esteja vazio
    if dados_numericos.empty:
        st.error("Nenhuma coluna numérica disponível para segmentação. Verifique os dados.")
        return pd.DataFrame()
    
    # Verifica se há valores NaN após o tratamento e interpolação
    if dados_numericos.isnull().values.any():
        st.error("Dados contêm valores ausentes após o tratamento. Verifique o processo de preparação dos dados.")
        return pd.DataFrame()
    
    # Segmentação usando KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dados['Etapa_Fabricacao_ID'] = kmeans.fit_predict(dados_numericos)
    dados['Etapa_Fabricacao'] = dados['Etapa_Fabricacao_ID'].map(ETAPAS_NOMES)
    dados = dados.sort_values(by=['Date/Time', 'Etapa_Fabricacao']).reset_index(drop=True)
    return dados

# Suavização alternativa com média móvel
def suavizar_com_media_movel(df, window=3):
    if df.empty:
        st.warning("Nenhum dado disponível para suavização.")
        return pd.DataFrame()

    df_suave = df.copy()
    for coluna in df.select_dtypes(include=[np.number]).columns:
        df_suave[coluna] = df[coluna].rolling(window=window, min_periods=1).mean()
    return df_suave

# Preparar dados para entrada no modelo LSTM
def preparar_dados_para_modelo(dados):
    if dados.empty:
        st.warning("Nenhum dado disponível para preparação do modelo.")
        return np.array([]), np.array([]), None

    scaler = MinMaxScaler()
    dados_scaled = scaler.fit_transform(dados.select_dtypes(include=[np.number]))

    X = []
    y = []
    for i in range(len(dados_scaled) - 1):
        X.append(dados_scaled[i])
        y.append(dados_scaled[i + 1][0])  # Assume que o alvo é a próxima observação da primeira coluna
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler  # Retorna arrays numpy

# Função principal do aplicativo
def main():
    if "login" not in st.session_state:
        st.session_state["login"] = False

    if not st.session_state["login"]:
        login()
    else:
        st.sidebar.title("Menu")
        opcao = st.sidebar.radio("Escolha uma opção:", ["MODELO PREDITIVO", "Formulação Matemática"])

        if opcao == "MODELO PREDITIVO":
            st.title("Modelo de Previsão da Cor Após Etapa de Resfriamento - Cerveja Amstel")

            video_url = "https://www.veed.io/view/31af47b8-7c73-468e-80da-e166c625d803?panel=share"
            st.video(video_url)

            etapa_crisp_dm("Etapa: Entendimento do Negócio", 
                           "Estudo do impacto do processo de fabricação na cor do produto.",
                           "Definir o problema e estabelecer um objetivo claro para a modelagem.")
            
            etapa_crisp_dm("Etapa: Entendimento dos Dados", 
                           "Carregamos e filtramos os dados para focar no produto AMSTL.", 
                           "Coletar dados necessários e garantir integridade para o produto AMSTL.")
            dados = carregar_dados()
            if dados.empty:
                st.stop()  # Interrompe a execução se os dados não forem carregados corretamente
            st.dataframe(dados)

            etapa_crisp_dm("Etapa: Preparação dos Dados", 
                           "Tratamento dos dados para lidar com ausências e outliers.", 
                           "Garantir a consistência e a qualidade dos dados para a modelagem.")
            dados_tratados = tratar_dados(dados)
            if dados_tratados.empty:
                st.stop()
            st.dataframe(dados_tratados)

            etapa_crisp_dm("Etapa: Preparação dos Dados", 
                           "Suavizamos os dados usando média móvel.", 
                           "Reduzir flutuações nos dados para melhorar a estabilidade da modelagem.")
            dados_suavizados = suavizar_com_media_movel(dados_tratados)
            if dados_suavizados.empty:
                st.stop()
            st.dataframe(dados_suavizados)

            etapa_crisp_dm("Etapa: Preparação dos Dados", 
                           "Segmentação das etapas de fabricação com KMeans.", 
                           "Identificar fases distintas do processo de fabricação para modelagem.")
            dados_segmentados = segmentar_etapas(dados_suavizados)
            if dados_segmentados.empty:
                st.stop()
            st.dataframe(dados_segmentados)

            etapa_crisp_dm("Etapa: Modelagem", 
                           "Construção de modelo preditivo LSTM para previsão da cor.",
                           "Prever a cor do produto após a etapa de resfriamento.")
            X, y, scaler = preparar_dados_para_modelo(dados_segmentados)
            if X.size == 0 or y.size == 0:
                st.stop()

            # (continuação do treinamento e avaliação do modelo...)

# Executar o aplicativo
if __name__ == "__main__":
    main()
