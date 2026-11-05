"""
TCC — Validação da inferência INT8 (equivalência Python x firmware)
Arquivo: tcc_validar_inferencia_int8.py
Roda DEPOIS do tcc_exportar_tflite_3classes.py (precisa de model.h, detector_surtos_int8.tflite e o dataset).

POR QUE EXISTE:
  Sem a placa, a pergunta é: "o que o ESP32 calcularia dá a mesma resposta que o Python?"
  Este script lê o model.h (o mesmo arquivo que vai para o firmware), refaz a normalização
  e a quantização do jeito que o C faz, e compara com o modelo .tflite.

O QUE VERIFICA:
  1. Os bytes do model.h são idênticos ao arquivo .tflite
  2. A inferência refeita "à moda do C" concorda com o interpretador TFLite
  3. Arredondar ou truncar na quantização muda o resultado? (erro comum no firmware)
  4. Quantos valores de entrada saturam na normalização
"""
import re, numpy as np, tensorflow as tf
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings('ignore')

SEED = 42
NOMES = ['Normal', 'Surto Leve', 'Surto Severo']

# ── 1. Lê o model.h ───────────────────────────────────────────────────────────
h = open('model.h').read()
bloco = h[h.index('modelo_surtos[]'):h.index('};')]
bytes_h = bytes(int(b, 16) for b in re.findall(r'0x([0-9a-fA-F]{2})', bloco))
vet = lambda nome: np.array([float(v) for v in re.search(nome + r'\[N_FEATURES\] = \{(.*?)\}', h).group(1).replace('f', '').split(',')], dtype=np.float32)
X_MIN, X_MAX = vet('X_MIN'), vet('X_MAX')
SCALE = np.float32(re.search(r'INPUT_SCALE = ([^f;]+)f', h).group(1))
ZP = int(re.search(r'INPUT_ZERO_POINT = (-?\d+)', h).group(1))

tflite_arq = open('detector_surtos_int8.tflite', 'rb').read()
print("[1] Bytes do model.h == arquivo .tflite:", "✓ IDÊNTICOS" if bytes_h == tflite_arq else "✗ DIFERENTES")
print(f"    Tamanho: {len(bytes_h)} bytes | INPUT_SCALE={SCALE:.6f} | ZERO_POINT={ZP}")

# ── Dados de teste (mesma divisão do script de exportação) ────────────────────
d = np.load('dados_augmentados_3classes.npz', allow_pickle=True)
X_raw, y, t = d['X'], d['y'], d['tempo']
DT = float(t[1] - t[0])

def extrair_features(s):
    n = len(s)
    rms = np.sqrt(np.mean(s**2)); pico = np.max(np.abs(s)); fc = pico / (rms + 1e-9); std = np.std(s)
    esp = np.abs(fft(s))[:n//2]; f = fftfreq(n, d=DT)[:n//2]; E = np.sum(esp**2) + 1e-9
    prop_hf = np.sum(esp[f > 100e3]**2) / E; cen = np.sum(f * esp**2) / E
    m = f > 10e3; fd = f[m][np.argmax(esp[m])] if np.any(m) else 0.0
    p = esp / (np.sum(esp) + 1e-9); ent = -np.sum(p * np.log(p + 1e-9))
    return np.array([rms, pico, fc, std, prop_hf, cen, fd, ent], dtype=np.float32)

X = np.array([extrair_features(s) for s in X_raw])
_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# ── 2. Interpretador TFLite usando os bytes do model.h ────────────────────────
interp = tf.lite.Interpreter(model_content=bytes_h); interp.allocate_tensors()
i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]

def classificar(q):
    interp.set_tensor(i_det['index'], q.reshape(1, -1).astype(np.int8)); interp.invoke()
    return int(interp.get_tensor(o_det['index'])[0].argmax())

def normalizar_c(x):   # igual ao C: (x - MIN) / (MAX - MIN + 1e-9f), depois clamp [0,1], em float32
    return np.clip((x - X_MIN) / (X_MAX - X_MIN + np.float32(1e-9)), 0.0, 1.0).astype(np.float32)

def quant_arredonda(xn):  # roundf() e saturação em int8
    return np.clip(np.round(xn / SCALE + ZP), -128, 127).astype(np.int8)

def quant_trunca(xn):     # cast direto (int8_t)(...), que trunca em direção a zero
    return np.clip(np.trunc(xn / SCALE + ZP), -128, 127).astype(np.int8)

xn_te = np.array([normalizar_c(x) for x in X_te])
p_arr = np.array([classificar(quant_arredonda(x)) for x in xn_te])
p_tru = np.array([classificar(quant_trunca(x)) for x in xn_te])

# Interpretador independente, carregado do arquivo .tflite
it2 = tf.lite.Interpreter(model_content=tflite_arq); it2.allocate_tensors()
a, b = it2.get_input_details()[0], it2.get_output_details()[0]
def clas2(q):
    it2.set_tensor(a['index'], q.reshape(1, -1).astype(np.int8)); it2.invoke()
    return int(it2.get_tensor(b['index'])[0].argmax())
p_arq = np.array([clas2(quant_arredonda(x)) for x in xn_te])

print(f"\n[2] Inferência estilo C (arredonda) x interpretador do .tflite: {(p_arr == p_arq).mean()*100:.1f}% de concordância")
print(f"[3] Acurácia no teste ({len(y_te)} sinais):")
print(f"      arredondando (roundf):   {(p_arr == y_te).mean()*100:.1f}%")
print(f"      truncando ((int8_t)...): {(p_tru == y_te).mean()*100:.1f}%   | difere do correto em {(p_tru != p_arr).sum()} previsões")

# ── 4. Saturação na normalização ──────────────────────────────────────────────
bruto = (X_te - X_MIN) / (X_MAX - X_MIN + np.float32(1e-9))
fora = ((bruto < 0) | (bruto > 1)).sum()
print(f"\n[4] Valores de entrada fora de [0,1] (saturados no clamp): {fora} de {bruto.size} ({fora/bruto.size*100:.2f}%)")
for i, nome in enumerate(['RMS','Pico','Fator_Crista','Desvio_Padrao','Prop_HF','Centroide','Freq_Dom','Entropia']):
    n_f = ((bruto[:, i] < 0) | (bruto[:, i] > 1)).sum()
    if n_f: print(f"      {nome}: {n_f} fora da faixa de treino")

ok = (bytes_h == tflite_arq) and (p_arr == p_arq).all()
print("\nCONCLUSÃO:", "✓ model.h e inferência equivalentes ao .tflite" if ok else "✗ há divergência, revisar")
print("Limite: isto valida o modelo e a aritmética de quantização, NÃO a extração de features em C nem o ADC.")
