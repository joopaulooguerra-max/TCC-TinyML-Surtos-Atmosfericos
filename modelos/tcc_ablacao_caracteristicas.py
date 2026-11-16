"""
Teste de ablação das características (Seção Resultados do TCC).

Pergunta: as características espectrais acrescentam algo às temporais?
Método: Random Forest (50 árvores, profundidade 8, folha mínima 4), validação
cruzada estratificada de 5 partes (semente 42), com três subconjuntos:
  - 8 características | só as 4 de tempo | só as 4 de frequência.
Também mede a fração da energia do impulso puro acima de 100 kHz.

Uso: python tcc_ablacao_caracteristicas.py [caminho/dados_augmentados_3classes.npz]
Resultado de referência: 8 = 97,8 ± 0,62 | tempo = 97,7 ± 0,44 | frequência = 83,4 ± 0,83
                         energia do impulso acima de 100 kHz = 0,49 %
"""
import sys
import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

ALFA, BETA = 14286.0, 2.16e6
caminho = sys.argv[1] if len(sys.argv) > 1 else 'dados_augmentados_3classes.npz'
d = np.load(caminho)
X, y, t = d['X'], d['y'], d['tempo']
DT = t[1] - t[0]

def features(s):
    n = len(s)
    rms = np.sqrt(np.mean(s ** 2)); pico = np.max(np.abs(s))
    fc = pico / (rms + 1e-9); sd = np.std(s)
    e = np.abs(fft(s))[:n // 2]; fr = fftfreq(n, d=DT)[:n // 2]
    E = np.sum(e ** 2) + 1e-9
    hf = np.sum(e[fr > 1e5] ** 2) / E
    cen = np.sum(fr * e ** 2) / E
    m = fr > 1e4
    fd = fr[m][np.argmax(e[m])] if m.any() else 0.0
    p = e / (np.sum(e) + 1e-9)
    ent = -np.sum(p * np.log(p + 1e-9))
    return [rms, pico, fc, sd, hf, cen, fd, ent]

F = np.array([features(s) for s in X])
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def acc(cols):
    rf = RandomForestClassifier(50, max_depth=8, min_samples_leaf=4, random_state=42)
    s = cross_val_score(rf, F[:, cols], y, cv=cv)
    return f"{100 * s.mean():.1f} ± {100 * s.std():.2f}"

print("8 características      :", acc(list(range(8))))
print("só tempo (4)           :", acc([0, 1, 2, 3]))
print("só frequência (4)      :", acc([4, 5, 6, 7]))

tt = np.arange(0, 200e-6, DT)
v = np.exp(-ALFA * tt) - np.exp(-BETA * tt)
e = np.abs(fft(v))[:len(tt) // 2] ** 2
fr = fftfreq(len(tt), DT)[:len(tt) // 2]
print("Energia do impulso puro acima de 100 kHz: %.2f %%" % (100 * e[fr > 1e5].sum() / e.sum()))
