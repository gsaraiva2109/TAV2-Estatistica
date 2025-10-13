import pandas as pd # pip install pandas
import seaborn as sns # pip install seaborn
import matplotlib.pyplot as plt # pip install matplotlib
import numpy as np # pip install numpy

def calcular_regressao_simples(x, y):
    x_media, y_media = x.mean(), y.mean()
    b = ((x - x_media) * (y - y_media)).sum() / ((x - x_media) ** 2).sum()
    a = y_media - b * x_media
    return a, b


def avaliar_modelo(df, x_col, y_col, a, b):
    x, y = df[x_col], df[y_col]
    y_previsto = a + b * x
    sqe = ((y - y_previsto) ** 2).sum()
    sqt = ((y - y.mean()) ** 2).sum()

    r_quadrado = 1 - (sqe / sqt)
    r = x.corr(y)
    return r_quadrado, r


def plotar_modelo(df, x_col, y_col, a, b, titulo):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, s=100, label='Dados Reais')

    x_vals = np.array([df[x_col].min(), df[x_col].max()])
    y_vals = a + b * x_vals
    plt.plot(x_vals, y_vals, color='red', lw=2, label=f'Reta de Regressão (ŷ = {a:.2f} + {b:.2f}x)')

    plt.title(titulo, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


def imprimir_relatorio(a, b, r2, r, x_label, y_label):
    print("--- Resultados da Análise de Regressão ---")
    print(f"Equação da Reta: {y_label}_Previsto = {a:.2f} + {b:.2f} * {x_label}")
    print(f"Coeficiente de Determinação (R²): {r2:.4f} ({r2:.2%})")
    print(f"Coeficiente de Correlação (r): {r:.4f}")
    if abs(r) > 0.8:
        print("-> A correlação é considerada muito forte.")
    elif abs(r) > 0.6:
        print("-> A correlação é considerada forte.")


def analisar_problema(problema):
    print("\n" + "=" * 60)
    print(f" EXECUTANDO ANÁLISE PARA: {problema['titulo'].upper()} ")
    print("=" * 60)

    df = pd.DataFrame(problema['dados'])
    x_col, y_col = problema['x_col'], problema['y_col']

    print("--- Análise Descritiva ---")
    print(df.describe())

    a, b = calcular_regressao_simples(df[x_col], df[y_col])
    r2, r = avaliar_modelo(df, x_col, y_col, a, b)

    imprimir_relatorio(a, b, r2, r, x_col, y_col)

    plotar_modelo(df, x_col, y_col, a, b, problema['titulo'])

    if 'previsoes' in problema:
        print("\n--- Previsões ---")
        for valor in problema['previsoes']:
            previsao = a + b * valor
            print(f"Para {x_col} = {valor}, a previsão de {y_col} é: {previsao:.2f}")


lista_de_problemas = [
    {
        'titulo': "Ride+: Distância vs. Valor da Corrida",
        'dados': {
            'Distancia_km': [2.0, 2.4, 2.8, 3.3, 3.7, 4.1, 4.5, 4.9, 5.4, 5.8, 6.2, 6.6, 7.1, 7.5, 7.9, 8.3, 8.7, 9.2,
                             9.6, 10.0],
            'Valor_Corrida_R$': [10.99, 10.34, 15.38, 20.78, 15.64, 17.54, 26.69, 25.33, 22.28, 28.22, 26.09, 27.98,
                                 32.70, 25.98, 28.63, 35.17, 35.26, 42.47, 39.47, 39.35]},
        'x_col': 'Distancia_km', 'y_col': 'Valor_Corrida_R$',
        'previsoes': [5, 7, 12]
    },
    {
        'titulo': "Algoritmo: Tamanho da Entrada vs. Tempo de Execução",
        'dados': {'Tamanho_Entrada_milhares': list(range(1, 21)),
                  'Tempo_Execucao_ms': [11, 20, 31, 40, 49, 62, 70, 81, 91, 101, 112, 121, 133, 140, 150, 160, 172, 180,
                                        191, 200]},
        'x_col': 'Tamanho_Entrada_milhares', 'y_col': 'Tempo_Execucao_ms'
    },
    {
        'titulo': "Datacenter: Temperatura vs. Consumo de Energia",
        'dados': {'Temperatura_C': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                  'Consumo_Energia_kWh': [510, 495, 480, 470, 460, 450, 440, 430, 420, 410, 405, 395, 385, 375, 370]},
        'x_col': 'Temperatura_C', 'y_col': 'Consumo_Energia_kWh',
        'previsoes': [20, 25, 30]
    }
]

for problema in lista_de_problemas:
    analisar_problema(problema)