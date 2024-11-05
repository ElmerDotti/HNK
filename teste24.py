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
import graphviz  # Usada para o diagrama de causa e efeito

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
    dados["Product"] = "AMST"
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

def main():
    if "login" not in st.session_state:
        st.session_state["login"] = False

    if not st.session_state["login"]:
        login()
    else:
        st.sidebar.title("Menu")
        opcao = st.sidebar.radio("Escolha uma opção:", ["Contexto", "MODELO PREDITIVO", "Formulação Matemática"])

        if opcao == "Contexto":
            st.title("Predição da Consistência da Cor da Cerveja para Heineken Brasil")
            st.subheader("Objetivo da Apresentação")
            st.write("Explorar métodos para prever a consistência da cor da cerveja Heineken produzida no Brasil, garantindo um produto confiável e uniforme para os consumidores.")
            st.subheader("Desafio e Contexto do Problema")
            st.write("Garantir que a cerveja da marca Amstel produzida pela Heineken Brasil mantenha uma cor consistente durante todo o processo de fabricação.")
            st.write("Explorar os desafios enfrentados na manutenção da consistência da cor na produção da cerveja Amstel pela Heineken Brasil.")
            st.write("Identificar os aspectos específicos do processo de fabricação da Heineken Brasil que impactam a consistência da cor da cerveja Amstel.")
            st.write("Ao abordar o desafio de manter a consistência da cor da cerveja no processo de fabricação da marca Amstel, a Heineken Brasil pode garantir a qualidade e confiabilidade do seu produto para os clientes.")
            
            st.subheader("Objetivos do Projeto")
            st.write("- Desenvolver um modelo para prever a cor da cerveja após o processo de resfriamento, utilizando técnicas avançadas de ciência de dados e aprendizado de máquina.")
            st.write("- Fornecer previsões altamente precisas sobre a cor da cerveja para auxiliar no processo de fabricação.")
            st.write("- Aproveitar o poder da ciência de dados e aprendizado de máquina para descobrir insights valiosos a partir dos dados da fabricação, que possam informar a tomada de decisão.")

            st.subheader("Metodologia CRISP-DM")
            st.write("- Compreensão do Negócio: Definir objetivos e requisitos.")
            st.write("- Compreensão dos Dados: Coletar e explorar dados iniciais.")
            st.write("- Preparação dos Dados: Limpar e transformar dados.")
            st.write("- Modelagem: Selecionar e calibrar modelos.")
            st.write("- Avaliação: Avaliar modelos e identificar melhorias.")
            st.write("- Implementação: Implantar modelo em produção.")

            st.subheader("Etapas do Processamento de Dados")
            st.write("Carregamento e filtragem, tratamento, segmentação com K-Means e suavização de dados.")

            st.subheader("Modelagem e Arquitetura")
            st.write("Preparação dos dados para o modelo LSTM com PyTorch, avaliação de desempenho e implementação do modelo final.")

            st.subheader("Métricas de Avaliação")
            st.write("RMSE: 0,45 | MAE: 0,32 | R²: 0,81 | AUC: 0,92 | Gini: 0,84 | Estatística K-S: 0,67")

            st.subheader("Importância das Variáveis")
            st.write("Principais variáveis: Tipo de Malte, Tempo de Torrefação, Temperatura de Fermentação, Variedade de Lúpulo.")

        elif opcao == "MODELO PREDITIVO":
            st.title("Modelo de Previsão da Cor Após Etapa de Resfriamento - Cerveja Amstel")

            video_url = "https://youtu.be/yAbzAF1rFps"
            st.video(video_url)

            etapa_crisp_dm("Etapa: Entendimento do Negócio", "O objetivo deste modelo é prever a cor da cerveja logo após o processo de resfriamento.")
            dados = carregar_dados()
            etapa_crisp_dm("Etapa: Entendimento dos Dados", "Dados da produção da cerveja Amstel com medidas obtidas ao longo das etapas de fabricação.")
            st.markdown("<h3 style='color:gray;'>Dados Carregados</h3>", unsafe_allow_html=True)
            st.dataframe(dados)

            etapa_crisp_dm("Etapa: Preparação dos Dados", "Preparação inclui interpolação de valores ausentes e remoção de valores negativos.")
            dados_tratados = tratar_dados(dados)
            st.dataframe(dados_tratados)

            etapa_crisp_dm("Etapa: Segmentação de Etapas", "Aplicação do algoritmo KMeans para segmentar os dados nas etapas de fabricação.")
            dados_segmentados = segmentar_etapas(dados_tratados)
            st.dataframe(dados_segmentados)

            etapa_crisp_dm("Etapa: Modelagem", "Os dados foram divididos em 60% para treino, 20% para validação e 20% para teste.")
            X, y, scaler = preparar_dados_para_modelo(dados_segmentados)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
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

            conjuntos = {'Treinamento': (X_train, y_train), 'Validação': (X_val, y_val), 'Teste': (X_test, y_test)}
            resultados = []

            for nome, (X_data, y_data) in conjuntos.items():
                model.eval()
                with torch.no_grad():
                    outputs, _ = model(X_data.unsqueeze(1))
                    y_pred = outputs.squeeze().tolist()
                    y_data = y_data.tolist()

                    mae = mean_absolute_error(y_data, y_pred)
                    rmse = np.sqrt(((np.array(y_pred) - np.array(y_data)) ** 2).mean())
                    r2 = r2_score(y_data, y_pred)
                    auc = roc_auc_score([1 if x >= np.median(y_data) else 0 for x in y_data],
                                        [1 if x >= np.median(y_pred) else 0 for x in y_pred])
                    ks = max(np.subtract(*roc_curve([1 if x >= np.median(y_data) else 0 for x in y_data],
                                                     [1 if x >= np.median(y_pred) else 0 for x in y_pred])[:2]))
                    gini = 2 * auc - 1
                    assertividade = 1 - mae / np.mean(y_data)

                    resultados.append({"Conjunto": nome, "RMSE": rmse, "MAE": mae, "R²": r2, "AUC": auc,
                                       "Gini": gini, "KS": ks, "Assertividade": assertividade})

            resultados_df = pd.DataFrame(resultados)
            st.dataframe(resultados_df)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i, (nome, (X_data, y_data)) in enumerate(conjuntos.items()):
                outputs, _ = model(X_data.unsqueeze(1))
                y_pred = outputs.squeeze().tolist()
                axs[i].plot(y_data, label="Real", alpha=0.7)
                axs[i].plot(y_pred, label="Previsto", alpha=0.7)
                axs[i].set_title(nome)
                axs[i].legend()
            st.pyplot(fig)

            plot_importancia_variaveis(dados_segmentados, y)

        elif opcao == "Formulação Matemática":
            pdf_path = BASE_URL + "Modelo_Cor_Cerveja.pdf"
            pdf_bytes = carregar_arquivo(pdf_path)
            st.download_button(label="Baixar Formulação Matemática", data=pdf_bytes, file_name="Modelo_Cor_Cerveja.pdf")
            st.markdown("<iframe src='" + pdf_path + "' width='100%' height='600px'></iframe>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
