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
import networkx as nx
from PIL import Image
import requests

BASE_URL = "https://raw.githubusercontent.com/ElmerDotti/HNK/main/"

# Função para carregar arquivos do GitHub
def carregar_arquivo(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

# Tela de login
def login():
    st.image(f"{BASE_URL}logo-removebg.png", width=150)
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

# Configuração de estilo para exibir as etapas do método CRISP-DM em fonte cinza
def etapa_crisp_dm(texto):
    st.markdown(f"<h5 style='color:gray;'>{texto}</h5>", unsafe_allow_html=True)

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
    dados = dados[dados['Product'] == 'AMST']
    dados['Date/Time'] = pd.to_datetime(dados['Date/Time'])
    dados = dados.drop(columns=['Product'])  # Remover coluna não numérica
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
    return X, y, scaler

# Definição da rede neural usando PyTorch com Dropout
class BeerColorPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(BeerColorPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out, h_n[-1]

# Função para criar e exibir o fluxograma do modelo usando networkx
def plot_fluxograma_modelo():
    etapas = list(ETAPAS_NOMES.values()) + ["Prever Cor"]
    G = nx.DiGraph()
    
    # Adiciona nós e arestas
    for i in range(len(etapas) - 1):
        G.add_edge(etapas[i], etapas[i + 1])
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 5))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray", arrows=True)
    nx.draw_networkx_nodes(G, pos, nodelist=["Prever Cor"], node_color="green")  # Destaca "Prever Cor" em verde
    plt.title("Fluxograma do Modelo de Previsão de Cor")
    st.pyplot(plt)

# Função para calcular e plotar a importância das variáveis usando a correlação
def plot_importancia_variaveis(dados, cor_prevista):
    dados_numericos = dados.select_dtypes(include=[np.number])
    correlacoes = dados_numericos.corrwith(pd.Series(cor_prevista)).abs()
    top_corr = correlacoes.sort_values(ascending=False).head(10)
    
    st.markdown("<h3 style='color:gray;'>Importância das Variáveis na Previsão de Cor</h3>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    top_corr.plot(kind='bar', ax=ax)
    ax.set_title("Importância das Variáveis")
    ax.set_ylabel("Correlação Absoluta")
    st.pyplot(fig)

    # Plot matriz de correlação para as 10 variáveis principais
    top_vars = dados_numericos[top_corr.index]
    corr_matrix = top_vars.corr()
    st.markdown("<h3 style='color:gray;'>Matriz de Correlação das Principais Variáveis</h3>", unsafe_allow_html=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

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

            video_url = "https://youtu.be/yAbzAF1rFps"
            st.video(video_url)

            etapa_crisp_dm("Etapa: Entendimento do Negócio")
            st.markdown("<p style='color:gray;'>O objetivo deste modelo é prever a cor da cerveja logo após o processo de resfriamento.</p>", unsafe_allow_html=True)

            etapa_crisp_dm("Etapa: Entendimento dos Dados")
            st.markdown("<h3 style='color:gray;'>Dados Carregados</h3>", unsafe_allow_html=True)
            dados = carregar_dados()
            st.dataframe(dados)

            etapa_crisp_dm("Etapa: Preparação dos Dados")
            st.markdown("<h3 style='color:gray;'>Tratamento de Dados</h3>", unsafe_allow_html=True)
            dados_tratados = tratar_dados(dados)
            st.dataframe(dados_tratados)

            etapa_crisp_dm("Etapa: Preparação dos Dados")
            st.markdown("<h3 style='color:gray;'>Dados com Segmentação de Etapas</h3>", unsafe_allow_html=True)
            dados_segmentados = segmentar_etapas(dados_tratados)
            st.dataframe(dados_segmentados)

            etapa_crisp_dm("Etapa: Modelagem")
            st.markdown("<h3 style='color:gray;'>Fluxograma do Modelo de Previsão</h3>", unsafe_allow_html=True)
            plot_fluxograma_modelo()

            etapa_crisp_dm("Etapa: Modelagem")
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

            etapa_crisp_dm("Etapa: Avaliação")
            plot_importancia_variaveis(dados_segmentados, y)

        elif opcao == "Formulação Matemática":
            pdf_path = BASE_URL + "Modelo_Cor_Cerveja.pdf"
            pdf_bytes = carregar_arquivo(pdf_path)
            st.download_button(label="Baixar Formulação Matemática", data=pdf_bytes, file_name="Modelo_Cor_Cerveja.pdf")
            st.markdown("<iframe src='" + pdf_path + "' width='100%' height='600px'></iframe>", unsafe_allow_html=True)

# Executar o aplicativo
if __name__ == "__main__":
    main()
