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
    caminho = BASE_URL + "Heineken%20-%20Data%20Science%20CB%20Use%20Case%202024.csv"
    dados = pd.read_csv(caminho)
    dados = dados[dados['Product'] == 'AMSTL']
    dados['Date/Time'] = pd.to_datetime(dados['Date/Time'])
    return dados

# Tratamento de dados: interpolação e valores absolutos para negativos
def tratar_dados(dados):
    dados = dados.copy()
    dados.interpolate(method='linear', inplace=True)
    dados.update(dados.select_dtypes(include=[np.number]).abs())
    return dados

# Segmentação das etapas de fabricação com KMeans
def segmentar_etapas(dados, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dados_numericos = dados.select_dtypes(include=[np.number])
    dados['Etapa_Fabricacao_ID'] = kmeans.fit_predict(dados_numericos)
    dados['Etapa_Fabricacao'] = dados['Etapa_Fabricacao_ID'].map(ETAPAS_NOMES)
    dados = dados.sort_values(by=['Date/Time', 'Etapa_Fabricacao']).reset_index(drop=True)
    return dados

# Suavização alternativa com média móvel
def suavizar_com_media_movel(df, window=3):
    df_suave = df.copy()
    for coluna in df.select_dtypes(include=[np.number]).columns:
        df_suave[coluna] = df[coluna].rolling(window=window, min_periods=1).mean()
    return df_suave

# Preparar dados para entrada no modelo LSTM
def preparar_dados_para_modelo(dados):
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

# Definição da rede neural usando PyTorch com Dropout
class BeerColorPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(BeerColorPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out, h_n[-1]  # Retorna saída e pesos ocultos

# Função para criar e exibir o fluxograma do modelo usando plotly
def plot_fluxograma_modelo():
    fig = go.Figure()

    # Adiciona os nós do fluxograma
    etapas = list(ETAPAS_NOMES.values()) + ["Prever Cor"]
    pos_x = [i * 2 for i in range(len(etapas))]
    pos_y = [1] * len(etapas)
    color_map = ["lightblue"] * (len(etapas) - 1) + ["green"]

    for i, etapa in enumerate(etapas):
        fig.add_trace(go.Scatter(
            x=[pos_x[i]], y=[pos_y[i]],
            mode="markers+text",
            marker=dict(size=30, color=color_map[i]),
            text=etapa,
            textposition="bottom center",
            showlegend=False
        ))

    # Adiciona as conexões
    for i in range(len(etapas) - 1):
        fig.add_trace(go.Scatter(
            x=[pos_x[i], pos_x[i+1]], y=[pos_y[i], pos_y[i+1]],
            mode="lines",
            line=dict(width=2, color="black"),
            showlegend=False
        ))

    fig.update_layout(
        title="Fluxograma do Modelo de Previsão de Cor",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig)

# Treinamento da rede neural com 5.000 iterações garantidas
def treinar_rede_neural(X_train, y_train, X_val, y_val, input_dim, output_dim):
    model = BeerColorPredictor(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5000):  # 5.000 iterações garantidas
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train.unsqueeze(1))
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(X_val.unsqueeze(1))
                val_loss = criterion(val_outputs.squeeze(), y_val)
                st.write(f'Época [{epoch+1}/5000], Loss: {loss.item():.4f}, Val_Loss: {val_loss.item():.4f}')
    
    return model

# Função para realizar backtest e calcular métricas de desempenho para treino, validação e teste
def realizar_backtest(model, X_data, y_data, conjunto):
    model.eval()
    with torch.no_grad():
        outputs, hidden_weights = model(X_data.unsqueeze(1))
        y_pred = outputs.squeeze().tolist()
        y_data = y_data.tolist()
        
        mae = mean_absolute_error(y_data, y_pred)
        rmse = np.sqrt(((np.array(y_pred) - np.array(y_data)) ** 2).mean())
        assertividade = 1 - mae / np.mean(y_data)
        r2 = r2_score(y_data, y_pred)

        median_value = np.median(y_data)
        y_data_bin = [1 if x >= median_value else 0 for x in y_data]
        y_pred_bin = [1 if x >= median_value else 0 for x in y_pred]
        
        auc = roc_auc_score(y_data_bin, y_pred_bin)
        fpr, tpr, _ = roc_curve(y_data_bin, y_pred_bin)
        ks = max(tpr - fpr)
        gini = 2 * auc - 1
        
    return y_data, y_pred, rmse, mae, assertividade, r2, auc, gini, ks

# Função para calcular e plotar a importância das variáveis usando a correlação
def plot_importancia_variaveis(dados, cor_prevista):
    dados_numericos = dados.select_dtypes(include=[np.number])
    correlacoes = dados_numericos.corrwith(pd.Series(cor_prevista)).abs()
    top_corr = correlacoes.sort_values(ascending=False).head(10)
    
    st.markdown("<h3 style='color:white;'>Importância das Variáveis na Previsão de Cor</h3>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    top_corr.plot(kind='bar', ax=ax)
    ax.set_title("Importância das Variáveis")
    ax.set_ylabel("Correlação Absoluta")
    st.pyplot(fig)

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
            st.dataframe(dados)

            etapa_crisp_dm("Etapa: Preparação dos Dados", 
                           "Tratamento dos dados para lidar com ausências e outliers.", 
                           "Garantir a consistência e a qualidade dos dados para a modelagem.")
            dados_tratados = tratar_dados(dados)
            st.dataframe(dados_tratados)

            etapa_crisp_dm("Etapa: Preparação dos Dados", 
                           "Suavizamos os dados usando média móvel.", 
                           "Reduzir flutuações nos dados para melhorar a estabilidade da modelagem.")
            dados_suavizados = suavizar_com_media_movel(dados_tratados)
            st.dataframe(dados_suavizados)

            etapa_crisp_dm("Etapa: Preparação dos Dados", 
                           "Segmentação das etapas de fabricação com KMeans.", 
                           "Identificar fases distintas do processo de fabricação para modelagem.")
            dados_segmentados = segmentar_etapas(dados_suavizados)
            st.dataframe(dados_segmentados)

            etapa_crisp_dm("Etapa: Modelagem", 
                           "Construção de modelo preditivo LSTM para previsão da cor.",
                           "Prever a cor do produto após a etapa de resfriamento.")
            plot_fluxograma_modelo()

            X, y, scaler = preparar_dados_para_modelo(dados_segmentados)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)

            input_dim = X_train.shape[1]
            output_dim = 1
            model = treinar_rede_neural(X_train, y_train, X_val, y_val, input_dim=input_dim, output_dim=output_dim)

            etapa_crisp_dm("Etapa: Avaliação", 
                           "Avaliação das métricas de desempenho do modelo.", 
                           "Calcular métricas de performance e interpretar a efetividade do modelo.")
            conjuntos = {'Treinamento': (X_train, y_train), 'Validação': (X_val, y_val), 'Teste': (X_test, y_test)}
            resultados = []

            cols = st.columns(3)
            for idx, (nome, (X_data, y_data)) in enumerate(conjuntos.items()):
                y_data_real, y_data_pred, rmse, mae, assertividade, r2, auc, gini, ks = realizar_backtest(model, X_data, y_data, nome)
                resultados.append({
                    "Conjunto": nome, "RMSE": rmse, "MAE": mae, "Assertividade": assertividade,
                    "R²": r2, "AUC": auc, "Gini": gini, "KS": ks
                })
                
                fig, ax = plt.subplots()
                ax.plot(y_data_real, label="Valor Real")
                ax.plot(y_data_pred, label="Previsão", linestyle='--')
                ax.set_title(f"{nome}")
                ax.legend()
                cols[idx].pyplot(fig)

            resultados_df = pd.DataFrame(resultados)
            st.dataframe(resultados_df)

            etapa_crisp_dm("Etapa: Avaliação", 
                           "Importância das variáveis na previsão da cor.",
                           "Avaliar a relevância das variáveis usadas no modelo preditivo.")
            plot_importancia_variaveis(dados_segmentados, y)

        elif opcao == "Formulação Matemática":
            pdf_path = BASE_URL + "Modelo_Cor_Cerveja.pdf"
            pdf_bytes = carregar_arquivo(pdf_path)
            st.download_button(label="Baixar Formulação Matemática", data=pdf_bytes, file_name="Modelo_Cor_Cerveja.pdf")
            st.markdown("<p style='color:white;'>Baixe o documento acima para consultar o modelo matemático detalhado.</p>", unsafe_allow_html=True)

# Executar o aplicativo
if __name__ == "__main__":
    main()
