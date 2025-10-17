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
    },
        {
        'id': 'problema4',
        'titulo': "Problema 4: Temperatura vs. Deformação da Pavimentação (Dados Sumários)",
        'dados': { 
            'Temperatura_F': [1.478/20, (143.2158 + 1.478)/20],
            'Deformacao_Pavimento': [12.75/20, (12.75 + 1083.67)/20]
        },
        'x_col': 'Temperatura_F', 
        'y_col': 'Deformacao_Pavimento',
        'x_range': [0.0, 85.0],
        'previsoes': [85.0],
        'tipo': 'problema4_sumario'
    },
    {
        'id': 'problema5',
        'titulo': "Problema 5: Horas de Uso Semanal vs. Dados Gerados",
        'dados': {
            'Horas_Uso_Semanal': [
                6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0,
                12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
                6.8, 7.2, 7.9, 8.7, 9.3, 10.1, 10.8, 11.6, 12.2, 12.9, 13.8, 14.4, 15.1, 15.8, 16.2,
                7.4, 8.2, 9.7, 10.9, 12.7, 13.3, 14.7, 15.3, 16.0,
                7.6, 8.4, 9.8, 11.2, 12.8, 14.1
            ],
            'Dados_Gerados_MB': [
                350, 370, 390, 420, 450, 470, 495, 510, 540, 565, 585, 610,
                640, 665, 690, 710, 740, 765, 790, 820,
                360, 380, 410, 460, 480, 520, 550, 600, 630, 655, 705, 735, 770, 805, 830,
                395, 445, 505, 560, 650, 675, 750, 780, 815,
                405, 455, 515, 575, 660, 720
            ]
        },
        'x_col': 'Horas_Uso_Semanal', 
        'y_col': 'Dados_Gerados_MB',
        'previsoes': [15.0],
        'tipo': 'problema5'
    },
    {
        'id': 'problema6',
        'titulo': "Problema 6: Horas de Uso do Celular vs. Mensagens Enviadas",
        'dados': {
            'Horas_Uso_Celular': [
                5.0, 5.3, 5.7, 6.0, 6.3, 6.5, 6.8, 7.0, 7.2, 7.4,
                7.7, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6,
                9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6
            ],
            'Mensagens_Enviadas': [
                60, 100, 130, 120, 180, 150, 200, 170, 250, 220,
                190, 300, 230, 270, 350, 300, 400, 320, 380, 280,
                420, 340, 450, 370, 500, 400, 470, 360, 520, 480
            ]
        },
        'x_col': 'Horas_Uso_Celular', 
        'y_col': 'Mensagens_Enviadas',
        'previsoes_custom': [2, 4, 6],
        'tipo': 'problema6'
    }
]



def calcular_regressao_simples(x, y):
    x_media, y_media = x.mean(), y.mean()
    denominador = ((x - x_media) ** 2).sum()
    if denominador == 0:
        return None, None
    b = ((x - x_media) * (y - y_media)).sum() / denominador
    a = y_media - b * x_media
    return a, b

def avaliar_modelo(df, x_col, y_col, a, b):
    x, y = df[x_col], df[y_col]
    
    if a is None or b is None:
        return 0.0, 0.0

    y_previsto = a + b * x
    sqe = ((y - y_previsto) ** 2).sum()
    sqt = ((y - y.mean()) ** 2).sum()

    r_quadrado = 0.0
    if sqt != 0:
        r_quadrado = 1 - (sqe / sqt)
        
    r = x.corr(y)
    return r_quadrado, r

def plotar_modelo(df, x_col, y_col, a, b, titulo, show_scatter=True, x_range=None):
    plt.figure(figsize=(10, 6))
    if show_scatter:
        sns.scatterplot(data=df, x=x_col, y=y_col, s=100, label='Dados Observados')

    if a is not None and b is not None: 
        if x_range:
            x_vals = np.array([x_range[0], x_range[1]])
        else:
            x_vals = np.array([df[x_col].min(), df[x_col].max()])
        y_vals = a + b * x_vals
        plt.plot(x_vals, y_vals, color='red', lw=2, label=f'Reta de Regressão (ŷ = {a:.2f} + {b:.2f}x)')

    plt.title(titulo, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def imprimir_relatorio(a, b, r2, r, x_label, y_label, tipo_problema=None):
    print("--- Resultados da Análise de Regressão ---")
    if a is not None and b is not None:
        print(f"Equação da Reta: {y_label}_Previsto = {a:.4f} + {b:.4f} * {x_label}")
        print(f"Coeficiente Angular (b): {b:.4f}")
        print(f"Coeficiente Linear (a): {a:.4f}")
    else:
        print("Não foi possível calcular os coeficientes de regressão.")
    
    print(f"Coeficiente de Determinação (R²): {r2:.4f} ({r2:.2%})")
    print(f"Coeficiente de Correlação (r): {r:.4f}")

    if abs(r) > 0.8:
        print("-> A correlação é considerada muito forte.")
    elif abs(r) > 0.6:
        print("-> A correlação é considerada forte.")
    elif abs(r) > 0.3:
        print("-> A correlação é considerada moderada.")
    else:
        print("-> A correlação é considerada fraca ou inexistente.")

    if tipo_problema == "problema5" or tipo_problema == "problema6":
        if r > 0:
            print(f"A correlação é positiva, indicando que à medida que {x_label} aumenta, {y_label} também tende a aumentar.")
        else:
            print(f"A correlação é negativa, indicando que à medida que {x_label} aumenta, {y_label} tende a diminuir.")

        print(f"\n--- Confiabilidade do Modelo (baseado no R²) ---")
        if r2 > 0.7:
            print(f"Com R² = {r2:.2%}, o modelo é considerado **confiável** para explicar a relação.")
        elif r2 > 0.4:
            print(f"Com R² = {r2:.2%}, o modelo tem uma confiabilidade **moderada**.")
        else:
            print(f"Com R² = {r2:.2%}, o modelo tem uma confiabilidade **baixa** para explicar a relação.")


def analisar_problema(problema):
    print("\n" + "=" * 60)
    print(f" EXECUTANDO ANÁLISE PARA: {problema['titulo'].upper()} ")
    print("=" * 60)

    df = pd.DataFrame(problema['dados'])
    x_col, y_col = problema['x_col'], problema['y_col']

    print("--- Análise Descritiva ---")
    print(df.describe())

    a, b = calcular_regressao_simples(df[x_col], df[y_col])

    if a is None or b is None:
        print("\nNão foi possível calcular os coeficientes de regressão. Verifique se a variável independente (X) possui variação.")
        r2, r = 0.0, 0.0
    else:
        r2, r = avaliar_modelo(df, x_col, y_col, a, b)

    imprimir_relatorio(a, b, r2, r, x_col, y_col, problema.get('tipo', None))

    if problema.get('id') == 'problema4':
        x_plot_min = problema['x_range'][0]
        x_plot_max = problema['x_range'][1]
        
        df_plot = pd.DataFrame({
            x_col: [x_plot_min, x_plot_max],
            y_col: [a + b * x_plot_min, a + b * x_plot_max] if a is not None and b is not None else [0,0]
        })
        plotar_modelo(df_plot, x_col, y_col, a, b, problema['titulo'], show_scatter=False, x_range=(x_plot_min, x_plot_max))
    else:
        plotar_modelo(df, x_col, y_col, a, b, problema['titulo'], show_scatter=True)

    if 'previsoes' in problema and a is not None and b is not None:
        print("\n--- Previsões ---")
        for valor in problema['previsoes']:
            previsao = a + b * valor
            print(f"Para {x_col} = {valor}, a previsão de {y_col} é: {previsao:.2f}")

    if problema.get('id') == 'problema5' and a is not None and b is not None:
        print(f"\n--- Interpretação do Coeficiente Angular (b) ---")
        print(f"O coeficiente angular (b = {b:.4f}) indica que, para cada aumento de 1 hora no uso semanal do aplicativo (X),")
        print(f"espera-se um aumento de {b:.2f} MB na quantidade de dados gerados (Y).")
        
    if problema.get('id') == 'problema6' and a is not None and b is not None:
        print(f"\n--- Interpretação do Coeficiente Angular (b) ---")
        print(f"O coeficiente angular (b = {b:.4f}) indica que, para cada aumento de 1 hora no tempo de uso diário do celular (X),")
        print(f"espera-se um aumento de aproximadamente {b:.2f} mensagens enviadas por dia (Y).")
        print(f"Em outras palavras, quanto mais tempo um estudante usa o celular, mais mensagens ele tende a enviar.")
        
        print("\n--- (d) Estimação de Mensagens para Diferentes Horas de Uso ---")
        horas_para_estimar = problema['previsoes_custom']
        for horas in horas_para_estimar:
            mensagens_estimadas_pred = a + b * horas
            mensagens_estimadas_pred = max(0, mensagens_estimadas_pred)
            print(f"Para {horas} horas de uso, o número estimado de mensagens é: {mensagens_estimadas_pred:.2f}")



for problema in lista_de_problemas:
    analisar_problema(problema)