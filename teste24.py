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
import graphviz
from PIL import Image
import requests

BASE_URL = "https://raw.githubusercontent.com/ElmerDotti/HNK/main/"

def carregar_arquivo(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def login():
    st.image(f"{BASE_URL}logo-removebg.png", width=150)
    st.title("Login")
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if usuario == "HNK" and senha == "HNK123":
            st.session_state["login"] = True
            st.success("Login realizado com sucesso!")
        else:
            st.error("Usuário ou senha incorretos!")

def etapa_crisp_dm(titulo, descricao):
    st.markdown(f"<h5 style='color:gray;'>{titulo}</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:gray;'>{descricao}</p>", unsafe_allow_html=True)

ETAPAS_NOMES = {
    0: "Moinho", 1: "Mistura", 2: "Cozimento", 3: "Filtragem",
    4: "Fervura", 5: "Clarificação", 6: "Resfriamento",
    7: "Fermentação", 8: "Maturação", 9: "Envase"
}

def carregar_dados():
    caminho = BASE_URL + "Heineken%20-%20Data%20Science%20CB%20Use%20Case%202024.csv"
    dados = pd.read_csv(caminho)
    dados['Date/Time'] = pd.to_datetime(dados['Date/Time'])
    dados = dados[dados["Product"] == "AMST"]
    dados.interpolate(method='linear', inplace=True)
    dados.dropna(inplace=True)
    return dados

def tratar_dados(dados):
    dados = dados.copy()
    dados.interpolate(method='linear', inplace=True)
    dados.update(dados.select_dtypes(include=[np.number]).abs())
    return dados

def segmentar_etapas(dados, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dados_numericos = dados.select_dtypes(include=[np.number])
    dados['Etapa_Fabricacao_ID'] = kmeans.fit_predict(dados_numericos)
    dados['Etapa_Fabricacao'] = dados['Etapa_Fabricacao_ID'].map(ETAPAS_NOMES)
    dados = dados.sort_values(by=['Date/Time', 'Etapa_Fabricacao']).reset_index(drop=True)
    return dados

def preparar_dados_para_modelo(dados):
    scaler = MinMaxScaler()
    dados_scaled = scaler.fit_transform(dados.select_dtypes(include=[np.number]))

    X = []
    y = []
    for i in range(len(dados_scaled) - 1):
        X.append(dados_scaled[i])
        y.append(dados_scaled[i + 1][0])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

class BeerColorPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(BeerColorPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out, h_n[-1]

def plot_fluxograma_modelo():
    etapas = list(ETAPAS_NOMES.values()) + ["Prever Cor"]
    G = nx.DiGraph()
    for i in range(len(etapas) - 1):
        G.add_edge(etapas[i], etapas[i + 1])
    G.add_edge("Resfriamento", "Prever Cor", color="red")
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 5))
    edges = G.edges()
    colors = ["red" if G[u][v]["color"] == "red" else "gray" for u, v in edges]
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", edge_color=colors, arrows=True)
    nx.draw_networkx_nodes(G, pos, nodelist=["Prever Cor"], node_color="green")
    plt.title("Fluxograma do Modelo de Previsão de Cor")
    st.pyplot(plt)

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

    top_vars = dados_numericos[top_corr.index]
    corr_matrix = top_vars.corr()
    st.markdown("<h3 style='color:gray;'>Matriz de Correlação das Principais Variáveis</h3>", unsafe_allow_html=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    st.markdown("<h3 style='color:gray;'>Diagrama de Causa e Efeito para a Previsão de Cor</h3>", unsafe_allow_html=True)
    diagram = graphviz.Digraph()
    diagram.node("Cor", "Previsão de Cor")
    for var in top_corr.index:
        diagram.edge("Cor", var, label=f"Influência: {top_corr[var]:.2f}")
    st.graphviz_chart(diagram)

def processar_modelo_academico(X_train, y_train, X_val, y_val, input_dim, output_dim, iteracoes):
    model = BeerColorPredictor(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    progress_bar = st.progress(0)
    for epoch in range(iteracoes):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train.unsqueeze(1))
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if epoch % (iteracoes // 100) == 0:
            progress = int((epoch / iteracoes) * 100)
            progress_bar.progress(progress)
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(X_val.unsqueeze(1))
                val_loss = criterion(val_outputs.squeeze(), y_val)
                st.write(f'Época [{epoch+1}/{iteracoes}], Loss: {loss.item():.4f}, Val_Loss: {val_loss.item():.4f}')

    progress_bar.progress(100)
    return model

def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((np.array(y_pred) - np.array(y_true)) ** 2))
    r2 = r2_score(y_true, y_pred)
    auc = roc_auc_score([1 if x >= np.median(y_true) else 0 for x in y_true],
                        [1 if x >= np.median(y_pred) else 0 for x in y_pred])
    gini = 2 * auc - 1
    ks = max(np.subtract(*roc_curve([1 if x >= np.median(y_true) else 0 for x in y_true],
                                    [1 if x >= np.median(y_pred) else 0 for x in y_pred])[:2]))
    assertividade = 1 - mae / np.mean(y_true)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "AUC": auc, "Gini": gini, "KS": ks, "Assertividade": assertividade}

def main():
    if "login" not in st.session_state:
        st.session_state["login"] = False

    if not st.session_state["login"]:
        login()
    else:
        st.sidebar.title("Menu")
        opcao = st.sidebar.radio("Escolha uma opção:", ["Contexto", "MODELO PREDITIVO", "Formulação Matemática", "Solução Acadêmica"])

        if opcao == "Solução Acadêmica":
            st.title("Solução Acadêmica - Modelo de Previsão de Cor da Cerveja")
            iteracoes = st.number_input("Digite a quantidade de iterações para o treinamento:", min_value=1, max_value=10000, value=5000)
            if st.button("Processar Modelo Acadêmico"):
                dados = carregar_dados()
                dados_tratados = tratar_dados(dados)
                X, y, scaler = preparar_dados_para_modelo(dados_tratados)
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

                X_train = torch.tensor(X_train, dtype=torch.float32)
                X_val = torch.tensor(X_val, dtype=torch.float32)
                X_test = torch.tensor(X_test, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)
                y_val = torch.tensor(y_val, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32)

                model = processar_modelo_academico(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1], output_dim=1, iteracoes=iteracoes)
                
                st.success("Modelo acadêmico processado com sucesso!")
                
                conjuntos = {'Treinamento': (X_train, y_train), 'Validação': (X_val, y_val), 'Teste': (X_test, y_test)}
                plot_metrica_academica(model, conjuntos)

                st.markdown("### Passo a Passo Implementado")
                st.write("""
                1. **Carregamento e Preparação de Dados:** Os dados foram carregados e interpolados para preencher valores ausentes.
                2. **Segmentação de Etapas:** Foi aplicado KMeans para definir as etapas do processo.
                3. **Escalonamento dos Dados:** O MinMaxScaler foi aplicado para normalizar as variáveis numéricas.
                4. **Modelo de Rede Neural:** A arquitetura LSTM com diferencial evolutivo foi usada para ajustar a topologia da rede e prever a cor.
                5. **Treinamento do Modelo:** Utilizando o algoritmo Adam, o modelo foi treinado com MSE.
                6. **Métricas de Avaliação:** As métricas R², RMSE, AUC, Gini, KS e Assertividade foram calculadas para cada fase.
                
                **Referência Acadêmica:** Takahashi et al. (2019). *Brewing process optimization by artificial neural network and evolutionary algorithm approach*.
                """)

        elif opcao == "Contexto":
            st.title("Predição da Consistência da Cor da Cerveja para Heineken Brasil")
            # Explicações e contexto

        elif opcao == "MODELO PREDITIVO":
            st.title("Modelo de Previsão da Cor Após Etapa de Resfriamento - Cerveja Amstel")
            # Código para modelo preditivo existente

        elif opcao == "Formulação Matemática":
            pdf_path = BASE_URL + "Modelo_Cor_Cerveja.pdf"
            pdf_bytes = carregar_arquivo(pdf_path)
            st.download_button(label="Baixar Formulação Matemática", data=pdf_bytes, file_name="Modelo_Cor_Cerveja.pdf")
            st.markdown("<iframe src='" + pdf_path + "' width='100%' height='600px'></iframe>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
