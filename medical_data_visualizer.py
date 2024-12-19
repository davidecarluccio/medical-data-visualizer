import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Carica il dataset
df = pd.read_csv('medical_examination.csv')

# 2: Aggiungi una colonna 'overweight'
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)

# 3: Normalizza i dati delle colonne 'cholesterol' e 'gluc'
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)

# 4: Funzione per disegnare il cat plot
def draw_cat_plot():
    # 5: Prepara i dati per il cat plot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6: Raggruppa i dati e calcola i totali
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7: Disegna il cat plot con Seaborn
    fig = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar', height=5, aspect=1
    ).fig

    # 8: Salva e restituisci la figura
    fig.savefig('catplot.png')
    return fig

# 10: Funzione per disegnare la heat map
def draw_heat_map():
    # 11: Pulisci i dati
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12: Calcola la matrice di correlazione
    corr = df_heat.corr()

    # 13: Crea una maschera per la heat map
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Disegna la heat map
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f", square=True, cmap='coolwarm',
        cbar_kws={'shrink': 0.5}, ax=ax
    )

    # 15: Salva e restituisci la figura
    fig.savefig('heatmap.png')
    return fig
