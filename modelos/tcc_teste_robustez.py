"""
TCC — Teste de robustez fora da distribuição de treino
Arquivo: tcc_teste_robustez.py
Roda DEPOIS do tcc_exportar_tflite_3classes.py (precisa de model.h, detector_surtos_int8.tflite e o dataset).

PERGUNTA: o modelo aprendeu algo do sinal ou só o jeito como o gerador cria os dados?
COMO: treina com os sinais do dataset e testa em sinais NOVOS, gerados com condições
      que não aparecem no treino. Se a acurácia cai muito, o modelo depende do gerador.

Modelos avaliados (treinados com os mesmos 900 sinais):
  - Random Forest
  - Rede neural INT8 (a do model.h, a mesma que iria para o ESP32-S3)

Cenários (300 sinais cada, 100 por classe, sementes novas):
  A  Controle: mesma distribuição do treino
  B  Ruído maior (sigma 0,10 a 0,20; no treino era 0,01 a 0,06)
  C  Amplitude da senoide fora da faixa (0,5 a 0,8 ou 1,2 a 1,5; no treino 0,8 a 1,2)
  D  Impulso nas bordas da janela (no treino ficava entre 20% e 80%)
  E  Forma do impulso diferente (alfa e beta variando de 0,6x a 1,4x)
  F  Harmônicos de 3a e 5a ordem (não existem no treino)
  G  B + C + D + E juntos
"""
import re, numpy as np, tensorflow as tf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings('ignore')

SEED = 42
ALFA, BETA = 14286.0, 2.16e6
NOMES = ['Normal', 'Surto Leve', 'Surto Severo']

# ── Dados de treino e parâmetros do modelo exportado ─────────────────────────
d = np.load('dados_augmentados_3classes.npz', allow_pickle=True)
X_raw, y, t = d['X'], d['y'], d['tempo']
DT = float(t[1] - t[0]); N = len(t)

h = open('model.h').read()
bytes_h = bytes(int(b, 16) for b in re.findall(r'0x([0-9a-fA-F]{2})', h[h.index('modelo_surtos[]'):h.index('};')]))
vet = lambda nome: np.array([float(v) for v in re.search(nome + r'\[N_FEATURES\] = \{(.*?)\}', h).group(1).replace('f', '').split(',')], dtype=np.float32)
X_MIN, X_MAX = vet('X_MIN'), vet('X_MAX')
SCALE = np.float32(re.search(r'INPUT_SCALE = ([^f;]+)f', h).group(1))
ZP = int(re.search(r'INPUT_ZERO_POINT = (-?\d+)', h).group(1))
interp = tf.lite.Interpreter(model_content=bytes_h); interp.allocate_tensors()
i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]

def extrair_features(s):
    n = len(s)
    rms = np.sqrt(np.mean(s**2)); pico = np.max(np.abs(s)); fc = pico / (rms + 1e-9); std = np.std(s)
    esp = np.abs(fft(s))[:n//2]; f = fftfreq(n, d=DT)[:n//2]; E = np.sum(esp**2) + 1e-9
    prop_hf = np.sum(esp[f > 100e3]**2) / E; cen = np.sum(f * esp**2) / E
    m = f > 10e3; fd = f[m][np.argmax(esp[m])] if np.any(m) else 0.0
    p = esp / (np.sum(esp) + 1e-9); ent = -np.sum(p * np.log(p + 1e-9))
    return np.array([rms, pico, fc, std, prop_hf, cen, fd, ent], dtype=np.float32)

def prever_int8(X):
    xn = np.clip((X - X_MIN) / (X_MAX - X_MIN + np.float32(1e-9)), 0, 1).astype(np.float32)
    out = []
    for x in xn:
        q = np.clip(np.round(x / SCALE + ZP), -128, 127).astype(np.int8).reshape(1, -1)
        interp.set_tensor(i_det['index'], q); interp.invoke()
        out.append(interp.get_tensor(o_det['index'])[0].argmax())
    return np.array(out)

# ── Treino: mesma divisão do script de exportação (900 sinais) ───────────────
X = np.array([extrair_features(s) for s in X_raw])
X_tmp, _, y_tmp, _ = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_tr, _, y_tr, _ = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=SEED, stratify=y_tmp)
rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)

# ── Gerador parametrizado (padrão = o mesmo do tcc_augmentation_3classes.py) ─
def gerar(classe, rng, ruido=(0.01, 0.06), amp=(0.8, 1.2), pos=None, esc_alfa=(1, 1), esc_beta=(1, 1), harm=False):
    a = rng.uniform(*amp) if not isinstance(amp, list) else rng.uniform(*amp[rng.integers(len(amp))])
    fase = rng.uniform(0, 2*np.pi)
    s = a * np.sin(2*np.pi*60*t + fase) + rng.normal(0, rng.uniform(*ruido), N)
    if harm:
        for k in (3, 5):
            s += a * rng.uniform(0.05, 0.15) * np.sin(2*np.pi*60*k*t + rng.uniform(0, 2*np.pi))
    if classe == 0:
        return s
    vp = rng.uniform(1.5, 3.0) if classe == 1 else rng.uniform(3.0, 6.0)
    if pos is None:
        p0 = int(rng.integers(N//5, 4*N//5))
    else:
        faixa = pos[rng.integers(len(pos))]; p0 = int(rng.integers(*faixa))
    tl = t - t[p0]
    al = ALFA * rng.uniform(*esc_alfa); be = BETA * rng.uniform(*esc_beta)
    imp = vp * (np.exp(-al*tl) - np.exp(-be*tl)); imp[tl < 0] = 0
    imp = imp / (np.max(np.abs(imp)) + 1e-9) * vp
    return s + imp

CENARIOS = {
  'A  Controle (mesma distribuição)': dict(),
  'B  Ruído maior (0,10 a 0,20)': dict(ruido=(0.10, 0.20)),
  'C  Amplitude da senoide fora da faixa': dict(amp=[(0.5, 0.8), (1.2, 1.5)]),
  'D  Impulso nas bordas da janela': dict(pos=[(5, N//5), (4*N//5, N - 80)]),
  'E  Forma do impulso diferente (0,6x a 1,4x)': dict(esc_alfa=(0.6, 1.4), esc_beta=(0.6, 1.4)),
  'F  Harmônicos de 3a e 5a ordem': dict(harm=True),
  'G  B + C + D + E juntos': dict(ruido=(0.10, 0.20), amp=[(0.5, 0.8), (1.2, 1.5)], pos=[(5, N//5), (4*N//5, N - 80)], esc_alfa=(0.6, 1.4), esc_beta=(0.6, 1.4)),
}

res = {}; conf_G = {}
print("Treino: 900 sinais | Teste: 300 sinais novos por cenário (100 por classe)\n")
print(f"{'Cenário':<46}{'Random Forest':>15}{'INT8 (model.h)':>16}")
for i, (nome, par) in enumerate(CENARIOS.items()):
    rng = np.random.default_rng(1000 + i)
    ys = np.array([0]*100 + [1]*100 + [2]*100)
    Xs = np.array([extrair_features(gerar(c, rng, **par)) for c in ys])
    p_rf, p_q = rf.predict(Xs), prever_int8(Xs)
    res[nome] = ((p_rf == ys).mean() * 100, (p_q == ys).mean() * 100)
    if nome.startswith('G') or nome.startswith('B'):
        conf_G[nome[0]] = confusion_matrix(ys, p_q)
    print(f"{nome:<46}{res[nome][0]:>14.1f}%{res[nome][1]:>15.1f}%")

print("\nMatrizes de confusão do INT8 (linha = real, coluna = previsto):")
for k in ('B', 'G'):
    print(f"\nCenário {k}:\n{conf_G[k]}")

# ── Gráfico ──────────────────────────────────────────────────────────────────
nomes = list(res.keys()); idx = np.arange(len(nomes))
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(idx - 0.2, [res[n][0] for n in nomes], 0.4, label='Random Forest', color='#185FA5')
ax.bar(idx + 0.2, [res[n][1] for n in nomes], 0.4, label='Rede neural INT8', color='#BA7517')
ax.axhline(100/3, color='gray', ls='--', lw=1, label='Acaso (33,3%)')
ax.set_xticks(idx); ax.set_xticklabels([n[0] for n in nomes]); ax.set_ylim(0, 105)
ax.set_ylabel('Acurácia (%)'); ax.set_title('Teste de robustez — sinais com condições não vistas no treino')
ax.legend(); ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('robustez_resultados.png', dpi=150); plt.show()

# ── Registro em Markdown (para experimentos/) ────────────────────────────────
md = ['# Teste de robustez', '',
      'Treino com 900 sinais do dataset; teste com 300 sinais novos por cenário (100 por classe).', '',
      '| Cenário | Random Forest | INT8 |', '|---|:-:|:-:|']
md += [f'| {n} | {a:.1f}% | {b:.1f}% |' for n, (a, b) in res.items()]
open('robustez_resultados.md', 'w').write('\n'.join(md))
print("\nArquivos: robustez_resultados.png | robustez_resultados.md")
