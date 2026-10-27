"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TCC — Classificação de Sinais de Surtos Atmosféricos com TinyML            ║
║  Arquivo : tcc_pipeline_3classes_v2.py          (features corrigidas)       ║
║                                                                             ║
║  O QUE MUDOU EM RELAÇÃO À VERSÃO ANTERIOR:                                  ║
║  1. Mag_60Hz removida — era inválida (resolução FFT = 5kHz, 60Hz=bin 0)    ║
║     → substituída por Centroide Espectral                                   ║
║  2. Energia removida — redundante com RMS (Energia = RMS² × N)             ║
║     → substituída por Desvio Padrão                                         ║
║  3. Energia_HF: limiar de 500Hz → 100kHz e agora é uma PROPORÇÃO           ║
║     (relativa à energia total, não valor absoluto)                          ║
║  4. Freq_Dom: limiar ajustado de 100Hz → 10kHz                             ║
║                                                                             ║
║  RODE ESTE SCRIPT DEPOIS do tcc_augmentation_3classes.py                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — CARREGA OS DADOS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 58)
print("  TCC — Pipeline v2 — Features Corrigidas — 3 Classes")
print("=" * 58)

print("\n[1] Carregando dados...")
dados = np.load('dados_augmentados_3classes.npz', allow_pickle=True)
X_raw = dados['X']
y     = dados['y']
t     = dados['tempo']
DT    = float(t[1] - t[0])
FS    = 1.0 / DT     # taxa de amostragem real

print(f"    Sinais: {X_raw.shape[0]} janelas × {X_raw.shape[1]} amostras")
print(f"    Taxa de amostragem: {FS/1e6:.0f} MHz")
print(f"    Resolução da FFT:   {(FS/X_raw.shape[1])/1e3:.1f} kHz por bin")
print(f"    Classe 0 (Normal):       {(y==0).sum()} exemplos")
print(f"    Classe 1 (Surto Leve):   {(y==1).sum()} exemplos")
print(f"    Classe 2 (Surto Severo): {(y==2).sum()} exemplos")


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — EXTRAÇÃO DE CARACTERÍSTICAS (VERSÃO CORRIGIDA)
# ══════════════════════════════════════════════════════════════════════════════

NOMES_FEATURES = [
    'RMS',
    'Pico',
    'Fator_Crista',
    'Desvio_Padrao',    # NOVO — substitui Energia (que era redundante)
    'Prop_HF_100kHz',   # NOVO — proporção de energia acima de 100 kHz
    'Centroide',        # NOVO — substitui Mag_60Hz (que era inválida)
    'Freq_Dom_10kHz',   # AJUSTADO — limiar de 100 Hz → 10 kHz
    'Entropia',
]

LIMIAR_HF = 100e3   # 100 kHz — compatível com frente de 1,2 µs do surto
LIMIAR_FD = 10e3    # 10 kHz  — faixa relevante para nossa janela

def extrair_features(sinal):
    """
    Resume o sinal em 8 números que descrevem seu comportamento.

    Domínio do tempo:
      1. RMS          — raiz da média dos quadrados (energia média)
      2. Pico         — valor absoluto máximo
      3. Fator Crista — Pico / RMS  (sempre ~1,41 no normal, explode no surto)
      4. Desvio Padrão — dispersão dos valores em torno da média

    Domínio da frequência (após FFT):
      5. Prop_HF      — fração da energia total acima de 100 kHz
                        (a frente de 1,2 µs do surto cria componentes aqui)
      6. Centroide    — frequência média ponderada pela energia
                        (sobe em surtos — energia se espalha para frequências altas)
      7. Freq_Dom     — frequência com mais energia acima de 10 kHz
      8. Entropia     — quão espalhado está o espectro
                        (surtos aumentam o "caos" espectral)
    """
    n    = len(sinal)

    # ── Domínio do tempo ─────────────────────────────────────────────────────
    rms  = np.sqrt(np.mean(sinal ** 2))
    pico = np.max(np.abs(sinal))
    fc   = pico / (rms + 1e-9)
    std  = np.std(sinal)

    # ── Domínio da frequência ─────────────────────────────────────────────────
    esp  = np.abs(fft(sinal))[:n // 2]
    frqs = fftfreq(n, d=DT)[:n // 2]
    E_total = np.sum(esp ** 2) + 1e-9

    # Proporção de energia em alta frequência (>100 kHz)
    mask_hf  = frqs > LIMIAR_HF
    prop_hf  = np.sum(esp[mask_hf] ** 2) / E_total

    # Centroide espectral — "centro de gravidade" do espectro
    centroide = np.sum(frqs * esp ** 2) / E_total

    # Frequência dominante acima de 10 kHz
    mask_fd = frqs > LIMIAR_FD
    if np.any(mask_fd):
        fd = frqs[mask_fd][np.argmax(esp[mask_fd])]
    else:
        fd = 0.0

    # Entropia espectral
    en   = esp / (np.sum(esp) + 1e-9)
    entr = -np.sum(en * np.log(en + 1e-9))

    return np.array([rms, pico, fc, std,
                     prop_hf, centroide, fd, entr], dtype=np.float32)


print("\n[2] Extraindo features corrigidas — aguarde...")
X = np.array([extrair_features(s) for s in X_raw])
print(f"    Features extraídas: {X.shape}  →  8 números por sinal")
print(f"    Features: {NOMES_FEATURES}")


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — TREINO DO CLASSIFICADOR
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Treinando Random Forest — 3 classes...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
modelo.fit(X_tr, y_tr)
y_pred = modelo.predict(X_te)

NOMES_CLASSES = ['Normal', 'Surto Leve', 'Surto Severo']
print("\n── Relatório de Classificação ─────────────────────────────")
print(classification_report(y_te, y_pred, target_names=NOMES_CLASSES))


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — VALIDAÇÃO K-FOLD (k=5)
# ══════════════════════════════════════════════════════════════════════════════

print("── Validação K-Fold (k=5) ──────────────────────────────────")
kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X, y, cv=kf, scoring='accuracy')

print(f"  Acurácias por fold: {[f'{s:.3f}' for s in scores]}")
print(f"  Média:    {scores.mean():.4f}  ({scores.mean()*100:.1f}%)")
print(f"  Desvio:   {scores.std():.4f}")

if scores.mean() >= 0.95:
    print("  ✓ Resultado excelente — modelo consistente")
elif scores.mean() >= 0.85:
    print("  ✓ Resultado bom — adequado para TCC")
else:
    print("  ⚠ Abaixo do esperado — revisar features ou dados")


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 5 — GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'TCC — Resultados v2 — Features Corrigidas\n'
    'Random Forest | 1500 exemplos | K-Fold k=5',
    fontsize=13
)

# Matriz de confusão 3×3
ConfusionMatrixDisplay.from_predictions(
    y_te, y_pred,
    display_labels=NOMES_CLASSES,
    colorbar=False, ax=axes[0], cmap='Blues'
)
axes[0].set_title('Matriz de Confusão 3×3')

# Importância das features
imp   = modelo.feature_importances_
ordem = np.argsort(imp)
axes[1].barh(
    [NOMES_FEATURES[i] for i in ordem],
    imp[ordem],
    color=['#185FA5' if imp[i] >= 0.10 else '#AABDD0' for i in ordem]
)
axes[1].set_title('Importância das Features')
axes[1].set_xlabel('Importância relativa')
axes[1].axvline(x=1/8, color='gray', linestyle='--',
                alpha=0.5, label='Importância igual (12,5%)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('resultados_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[5] Gráfico salvo em 'resultados_v2.png'")


# ══════════════════════════════════════════════════════════════════════════════
# SALVAR MODELO E NORMALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

with open('modelo_3classes_v2.pkl', 'wb') as f:
    pickle.dump(modelo, f)

x_min = X.min(axis=0)
x_max = X.max(axis=0)
np.savez('normalizacao_3classes_v2.npz',
         x_min=x_min, x_max=x_max,
         features=np.array(NOMES_FEATURES))

print("[6] Arquivos salvos:")
print("    ├── modelo_3classes_v2.pkl")
print("    ├── normalizacao_3classes_v2.npz")
print("    └── resultados_v2.png")
print("\n✓ Pipeline v2 concluído!")
print("  Próximo: tcc_exportar_tflite_3classes.py")
