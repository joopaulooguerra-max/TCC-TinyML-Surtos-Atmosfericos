"""
TCC — Robustez com 10 sementes (média e desvio)
Arquivo: tcc_robustez_multisementes.py
Roda DEPOIS do tcc_exportar_tflite_3classes.py (precisa do dataset dados_augmentados_3classes.npz).

POR QUE EXISTE:
  Um único treino da rede neural não dá um número confiável: dois treinos da mesma rede
  deram 93% e 98,7% no mesmo cenário. Aqui a rede é treinada, quantizada para INT8 e avaliada
  10 vezes, com sementes diferentes, nos mesmos cenários do tcc_teste_robustez.py.
  A Random Forest também é repetida com 10 sementes, para a comparação ser justa.

O QUE NÃO MUDA: o model.h do firmware. Este script caracteriza o PROCEDIMENTO de treino,
  não um modelo específico.
"""
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.filterwarnings('ignore')

SPLIT_SEED = 42
SEMENTES = list(range(10))
ALFA, BETA = 14286.0, 2.16e6

d = np.load('dados_augmentados_3classes.npz', allow_pickle=True)
X_raw, y, t = d['X'], d['y'], d['tempo']
DT = float(t[1] - t[0]); N = len(t)

def extrair_features(s):
    n = len(s)
    rms = np.sqrt(np.mean(s**2)); pico = np.max(np.abs(s)); fc = pico / (rms + 1e-9); std = np.std(s)
    esp = np.abs(fft(s))[:n//2]; f = fftfreq(n, d=DT)[:n//2]; E = np.sum(esp**2) + 1e-9
    prop_hf = np.sum(esp[f > 100e3]**2) / E; cen = np.sum(f * esp**2) / E
    m = f > 10e3; fd = f[m][np.argmax(esp[m])] if np.any(m) else 0.0
    p = esp / (np.sum(esp) + 1e-9); ent = -np.sum(p * np.log(p + 1e-9))
    return np.array([rms, pico, fc, std, prop_hf, cen, fd, ent], dtype=np.float32)

# ── Gerador de cenários (igual ao tcc_teste_robustez.py) ─────────────────────
def gerar(classe, rng, ruido=(0.01, 0.06), amp=(0.8, 1.2), pos=None, esc_alfa=(1, 1), esc_beta=(1, 1), harm=False):
    a = rng.uniform(*amp) if not isinstance(amp, list) else rng.uniform(*amp[rng.integers(len(amp))])
    s = a * np.sin(2*np.pi*60*t + rng.uniform(0, 2*np.pi)) + rng.normal(0, rng.uniform(*ruido), N)
    if harm:
        for k in (3, 5):
            s += a * rng.uniform(0.05, 0.15) * np.sin(2*np.pi*60*k*t + rng.uniform(0, 2*np.pi))
    if classe == 0:
        return s
    vp = rng.uniform(1.5, 3.0) if classe == 1 else rng.uniform(3.0, 6.0)
    p0 = int(rng.integers(N//5, 4*N//5)) if pos is None else int(rng.integers(*pos[rng.integers(len(pos))]))
    tl = t - t[p0]
    al = ALFA * rng.uniform(*esc_alfa); be = BETA * rng.uniform(*esc_beta)
    imp = vp * (np.exp(-al*tl) - np.exp(-be*tl)); imp[tl < 0] = 0
    return s + imp / (np.max(np.abs(imp)) + 1e-9) * vp

BORDAS = [(5, N//5), (4*N//5, N - 80)]
CENARIOS = {
  'A Controle': dict(),
  'B Ruído maior': dict(ruido=(0.10, 0.20)),
  'C Amplitude fora': dict(amp=[(0.5, 0.8), (1.2, 1.5)]),
  'D Impulso nas bordas': dict(pos=BORDAS),
  'E Forma diferente': dict(esc_alfa=(0.6, 1.4), esc_beta=(0.6, 1.4)),
  'F Harmônicos': dict(harm=True),
  'G B+C+D+E': dict(ruido=(0.10, 0.20), amp=[(0.5, 0.8), (1.2, 1.5)], pos=BORDAS, esc_alfa=(0.6, 1.4), esc_beta=(0.6, 1.4)),
}
ys = np.array([0]*100 + [1]*100 + [2]*100)
print("Gerando os cenários de teste (uma vez, iguais para todas as sementes)...")
TESTES = {}
for i, (nome, par) in enumerate(CENARIOS.items()):
    rng = np.random.default_rng(1000 + i)
    TESTES[nome] = np.array([extrair_features(gerar(c, rng, **par)) for c in ys])

# ── Divisão 60/20/20 (fixa, igual à do script de exportação) ─────────────────
X = np.array([extrair_features(s) for s in X_raw])
X_tmp, _, y_tmp, _ = train_test_split(X, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=SPLIT_SEED, stratify=y_tmp)
x_min, x_max = X_tr.min(axis=0), X_tr.max(axis=0)
norm = lambda A: np.clip((A - x_min) / (x_max - x_min + 1e-9), 0.0, 1.0).astype(np.float32)
Xn_tr, Xn_va = norm(X_tr), norm(X_va)

def treinar_int8(semente):
    tf.keras.utils.set_random_seed(semente)
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(16, activation='relu'), tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m.fit(Xn_tr, y_tr, validation_data=(Xn_va, y_va), epochs=200, batch_size=32, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)])
    def calib():
        for i in range(min(200, len(Xn_tr))):
            yield [Xn_tr[i:i+1]]
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = calib
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8; c.inference_output_type = tf.int8
    return c.convert()

def prever_int8(tfl, X):
    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i['quantization']; out = []
    for x in norm(X):
        q = np.clip(np.round(x / s + z), -128, 127).astype(np.int8).reshape(1, -1)
        it.set_tensor(i['index'], q); it.invoke(); out.append(it.get_tensor(o['index'])[0].argmax())
    return np.array(out)

acc_q = {n: [] for n in CENARIOS}; acc_rf = {n: [] for n in CENARIOS}; tamanhos = []
print(f"\nTreinando e avaliando {len(SEMENTES)} sementes...")
for sem in SEMENTES:
    tfl = treinar_int8(sem); tamanhos.append(len(tfl))
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, random_state=sem, n_jobs=-1).fit(X_tr, y_tr)
    for nome, Xs in TESTES.items():
        acc_q[nome].append((prever_int8(tfl, Xs) == ys).mean() * 100)
        acc_rf[nome].append((rf.predict(Xs) == ys).mean() * 100)
    print(f"  semente {sem}: controle INT8 {acc_q['A Controle'][-1]:.1f}% | ruído maior {acc_q['B Ruído maior'][-1]:.1f}% | B+C+D+E {acc_q['G B+C+D+E'][-1]:.1f}%")

print(f"\nResultados em {len(SEMENTES)} sementes (média ± desvio | mínimo a máximo):\n")
print(f"{'Cenário':<24}{'Random Forest':<30}{'Rede INT8'}")
linhas = []
for n in CENARIOS:
    a, b = np.array(acc_rf[n]), np.array(acc_q[n])
    fa = f"{a.mean():.1f} ± {a.std():.1f}  ({a.min():.1f}-{a.max():.1f})"
    fb = f"{b.mean():.1f} ± {b.std():.1f}  ({b.min():.1f}-{b.max():.1f})"
    print(f"{n:<24}{fa:<30}{fb}"); linhas.append((n, fa, fb))
print(f"\nTamanho do modelo INT8: {np.mean(tamanhos)/1024:.2f} KB (varia de {min(tamanhos)} a {max(tamanhos)} bytes)")

nomes = list(CENARIOS)
fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(nomes))
ax.bar(x - 0.2, [np.mean(acc_rf[n]) for n in nomes], 0.4, yerr=[np.std(acc_rf[n]) for n in nomes], capsize=3, label='Random Forest', color='#185FA5')
ax.bar(x + 0.2, [np.mean(acc_q[n]) for n in nomes], 0.4, yerr=[np.std(acc_q[n]) for n in nomes], capsize=3, label='Rede INT8', color='#BA7517')
ax.axhline(100/3, color='gray', ls='--', lw=1, label='Acaso (33,3%)')
ax.set_xticks(x); ax.set_xticklabels([n.split()[0] for n in nomes]); ax.set_ylim(0, 105)
ax.set_ylabel('Acurácia (%)'); ax.set_title(f'Robustez: média ± desvio em {len(SEMENTES)} sementes'); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('robustez_multisementes.png', dpi=150); plt.show()

md = [f'# Robustez em {len(SEMENTES)} sementes', '',
      'Média ± desvio padrão (mínimo a máximo) da acurácia, em 300 sinais novos por cenário.', '',
      '| Cenário | Random Forest | Rede INT8 |', '|---|:-:|:-:|'] + [f'| {n} | {a} | {b} |' for n, a, b in linhas]
open('robustez_multisementes.md', 'w').write('\n'.join(md))
print("\nArquivos: robustez_multisementes.png | robustez_multisementes.md")
