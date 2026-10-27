"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TCC — Classificação de Sinais de Surtos Atmosféricos com TinyML            ║
║  Arquivo : tcc_augmentation_3classes.py                                     ║
║  O que faz: expande os 3 cenários da simulação em 1500 exemplos variados    ║
║             500 exemplos por classe, prontos para treinar o classificador   ║
║                                                                             ║
║  RODE ESTE SCRIPT DEPOIS do tcc_simulacao_v2.py                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

EXPLICAÇÃO SIMPLES DO QUE ACONTECE AQUI:
  A simulação gerou 3 formas de onda — uma Normal, uma Surto Leve e uma Surto
  Severo. Mas 3 exemplos são poucos para treinar qualquer modelo de ML.

  Este script pega os parâmetros físicos dessas simulações e cria 500 variações
  de cada uma, mudando coisas como:
    - A amplitude do sinal (rede pode estar em 95% ou 105% do nominal)
    - A fase da senoidal (o surto pode ocorrer em qualquer momento do ciclo)
    - O nível de ruído (às vezes a rede tem mais interferência)
    - A posição do surto na janela de tempo (pode ocorrer no início ou no fim)
    - A amplitude do pico (surto leve entre 1,5× e 3×, severo acima de 3×)

  No final teremos 1500 exemplos balanceados — 500 de cada classe.

Dependências: numpy (já presente no Google Colab)
"""

import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# PARÂMETROS — devem ser os mesmos da simulação v2
# ══════════════════════════════════════════════════════════════════════════════

DT        = 1e-7          # 100 ns — mesmo passo da simulação
T_TOTAL   = 200e-6        # 200 µs — mesma janela
t         = np.arange(0, T_TOTAL, DT)
N         = len(t)
FREQ_REDE = 60            # Hz
N_JANELAS = 500           # exemplos por classe
SEED      = 42
np.random.seed(SEED)

# Parâmetros do impulso CORRIGIDOS (mesmos do tcc_simulacao_v2.py)
ALFA = 14286.0
BETA = 2.16e6

print("=" * 55)
print("  TCC — Data Augmentation — 3 Classes")
print("=" * 55)
print(f"\n  Gerando {N_JANELAS} exemplos × 3 classes = {N_JANELAS*3} total")
print(f"  Amostras por janela: {N}\n")


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES GERADORAS
# ══════════════════════════════════════════════════════════════════════════════

def gerar_normal():
    """
    Classe 0 — Operação Normal.

    Gera uma senoidal de 60 Hz com:
    - Amplitude variando ±20% (tensão da rede nunca é perfeita)
    - Fase aleatória (podemos começar a medir em qualquer ponto do ciclo)
    - Ruído gaussiano variável (interferências do ambiente)
    """
    amp   = np.random.uniform(0.8, 1.2)        # ±20% de variação de amplitude
    fase  = np.random.uniform(0, 2*np.pi)       # fase aleatória
    ruido = np.random.uniform(0.01, 0.06)       # nível de ruído aleatório
    sinal = amp * np.sin(2*np.pi*FREQ_REDE*t + fase)
    return sinal + np.random.normal(0, ruido, N)


def gerar_surto_leve():
    """
    Classe 1 — Surto Leve.

    Mesma senoidal de 60 Hz, mas com um impulso de pico entre 1,5× e 3×
    a tensão nominal. Representa um raio que caiu longe da linha e induziu
    uma sobretensão moderada.

    O impulso pode aparecer em qualquer posição da janela de tempo
    (o raio não avisa quando vai cair).
    """
    amp   = np.random.uniform(0.8, 1.2)
    fase  = np.random.uniform(0, 2*np.pi)
    ruido = np.random.uniform(0.01, 0.06)
    sinal = amp * np.sin(2*np.pi*FREQ_REDE*t + fase)
    sinal += np.random.normal(0, ruido, N)

    # Impulso LEVE: pico entre 1,5× e 3× o nominal
    vpico   = np.random.uniform(1.5, 3.0)
    pos     = np.random.randint(N//5, 4*N//5)   # posição aleatória na janela
    t_local = t - t[pos]                         # tempo relativo ao surto
    impulso = vpico * (np.exp(-ALFA*t_local) - np.exp(-BETA*t_local))
    impulso[t_local < 0] = 0                     # surto só após o instante pos
    impulso = impulso / (np.max(np.abs(impulso)) + 1e-9) * vpico

    return sinal + impulso


def gerar_surto_severo():
    """
    Classe 2 — Surto Severo.

    Igual ao leve, mas com pico ACIMA de 3× o nominal.
    Representa uma descarga atmosférica direta ou muito próxima da linha.
    """
    amp   = np.random.uniform(0.8, 1.2)
    fase  = np.random.uniform(0, 2*np.pi)
    ruido = np.random.uniform(0.01, 0.06)
    sinal = amp * np.sin(2*np.pi*FREQ_REDE*t + fase)
    sinal += np.random.normal(0, ruido, N)

    # Impulso SEVERO: pico entre 3× e 6× o nominal
    vpico   = np.random.uniform(3.0, 6.0)
    pos     = np.random.randint(N//5, 4*N//5)
    t_local = t - t[pos]
    impulso = vpico * (np.exp(-ALFA*t_local) - np.exp(-BETA*t_local))
    impulso[t_local < 0] = 0
    impulso = impulso / (np.max(np.abs(impulso)) + 1e-9) * vpico

    return sinal + impulso


# ══════════════════════════════════════════════════════════════════════════════
# GERAÇÃO DO DATASET COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

print("[1] Gerando Classe 0 — Normal...")
X_normal = np.array([gerar_normal()       for _ in range(N_JANELAS)])
print(f"    ✓ {len(X_normal)} exemplos gerados")

print("[2] Gerando Classe 1 — Surto Leve...")
X_leve   = np.array([gerar_surto_leve()  for _ in range(N_JANELAS)])
print(f"    ✓ {len(X_leve)} exemplos gerados")

print("[3] Gerando Classe 2 — Surto Severo...")
X_severo = np.array([gerar_surto_severo() for _ in range(N_JANELAS)])
print(f"    ✓ {len(X_severo)} exemplos gerados")

# Junta tudo num dataset único
X = np.vstack([X_normal, X_leve, X_severo])
y = np.array([0]*N_JANELAS + [1]*N_JANELAS + [2]*N_JANELAS)

print(f"\n[4] Dataset final:")
print(f"    Shape: {X.shape}  →  {X.shape[0]} janelas × {X.shape[1]} amostras")
print(f"    Classe 0 (Normal):       {(y==0).sum()} exemplos")
print(f"    Classe 1 (Surto Leve):   {(y==1).sum()} exemplos")
print(f"    Classe 2 (Surto Severo): {(y==2).sum()} exemplos")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÃO — 1 exemplo de cada classe para conferência
# ══════════════════════════════════════════════════════════════════════════════

t_us = t * 1e6
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle('TCC — Exemplos do Dataset — 1 Amostra por Classe', fontsize=13)

cores  = ['steelblue', 'darkorange', 'crimson']
nomes  = ['Classe 0 — Normal', 'Classe 1 — Surto Leve', 'Classe 2 — Surto Severo']
dados  = [X_normal[0], X_leve[0], X_severo[0]]

for i, (ax, dado, nome, cor) in enumerate(zip(axes, dados, nomes, cores)):
    ax.plot(t_us, dado, color=cor, lw=0.8)
    ax.set_title(nome)
    ax.set_ylabel('Amplitude (p.u.)')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Tempo (µs)')
plt.tight_layout()
plt.savefig('exemplos_3classes.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[5] Gráfico salvo em 'exemplos_3classes.png'")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

np.savez('dados_augmentados_3classes.npz',
    X=X, y=y, tempo=t,
    classes=np.array(['Normal', 'Surto Leve', 'Surto Severo'])
)

print("[6] Dataset salvo em 'dados_augmentados_3classes.npz'")
print("\n✓ Passo 2 concluído!")
print("  Próximo passo: rode tcc_pipeline_3classes.py para treinar o modelo")
