"""
TCC — Exportação TFLite INT8 (3 classes) para ESP32-S3
Arquivo: tcc_exportar_tflite_3classes.py
Roda DEPOIS do tcc_augmentation_3classes.py (precisa de dados_augmentados_3classes.npz)

O QUE FAZ, EM ORDEM:
  1. Carrega os 1500 sinais e extrai as 8 features (mesma função do pipeline v2)
  2. Divide em treino (60%), validação (20%) e teste (20%)
  3. Normaliza com min/max calculado SÓ no treino (evita vazamento de dados)
  4. Treina uma rede neural pequena (8 -> 16 -> 8 -> 3, softmax)
  5. Converte para TFLite INT8 e mede a acurácia do modelo convertido
  6. Gera model.h (array C + constantes de normalização) para o firmware

Por que uma rede neural e não a Random Forest?
  A Random Forest não converte direto para TensorFlow Lite. A MLP converte,
  é pequena e roda no TFLite Micro. Comparamos as duas no mesmo teste.
"""
import numpy as np
import tensorflow as tf
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
NOMES_CLASSES = ['Normal', 'Surto Leve', 'Surto Severo']
NOMES_FEAT = ['RMS','Pico','Fator_Crista','Desvio_Padrao',
              'Prop_HF_100kHz','Centroide','Freq_Dom_10kHz','Entropia']

# ── 1. Dados e features (idêntico ao tcc_pipeline_3classes_v2.py) ─────────────
d = np.load('dados_augmentados_3classes.npz', allow_pickle=True)
X_raw, y, t = d['X'], d['y'], d['tempo']
DT = float(t[1] - t[0])

def extrair_features(s):
    n = len(s)
    rms = np.sqrt(np.mean(s**2)); pico = np.max(np.abs(s))
    fc = pico / (rms + 1e-9);     std = np.std(s)
    esp = np.abs(fft(s))[:n//2];  f = fftfreq(n, d=DT)[:n//2]
    E = np.sum(esp**2) + 1e-9
    prop_hf = np.sum(esp[f > 100e3]**2) / E
    centroide = np.sum(f * esp**2) / E
    m = f > 10e3
    fd = f[m][np.argmax(esp[m])] if np.any(m) else 0.0
    p = esp / (np.sum(esp) + 1e-9)
    entr = -np.sum(p * np.log(p + 1e-9))
    return np.array([rms, pico, fc, std, prop_hf, centroide, fd, entr], dtype=np.float32)

print("[1] Extraindo features...")
X = np.array([extrair_features(s) for s in X_raw])

# ── 2. Treino / validação / teste ─────────────────────────────────────────────
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=SEED, stratify=y_tmp)
print(f"[2] Treino {len(X_tr)} | Validação {len(X_va)} | Teste {len(X_te)}")

# ── 3. Normalização min-max calculada SÓ no treino ────────────────────────────
x_min, x_max = X_tr.min(axis=0), X_tr.max(axis=0)
norm = lambda A: np.clip((A - x_min) / (x_max - x_min + 1e-9), 0.0, 1.0).astype(np.float32)
Xn_tr, Xn_va, Xn_te = norm(X_tr), norm(X_va), norm(X_te)

# ── Referência: Random Forest no mesmo conjunto de teste ──────────────────────
rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4,
                            random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
acc_rf = (rf.predict(X_te) == y_te).mean()

# ── 4. Rede neural pequena ────────────────────────────────────────────────────
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
])
modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelo.fit(Xn_tr, y_tr, validation_data=(Xn_va, y_va), epochs=200, batch_size=32, verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                                       restore_best_weights=True)])
acc_nn = (modelo.predict(Xn_te, verbose=0).argmax(1) == y_te).mean()
print(f"[4] MLP float32 — acurácia no teste: {acc_nn*100:.1f}%  (Random Forest: {acc_rf*100:.1f}%)")

# ── 5. Conversão TFLite INT8 ──────────────────────────────────────────────────
def calibracao():
    for i in range(min(200, len(Xn_tr))):
        yield [Xn_tr[i:i+1]]

conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = calibracao
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite = conv.convert()
open('detector_surtos_int8.tflite', 'wb').write(tflite)

# Mede a acurácia do modelo JÁ convertido (é ele que vai para o ESP32)
interp = tf.lite.Interpreter(model_content=tflite); interp.allocate_tensors()
i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
s_in, z_in = i_det['quantization']
pred = []
for x in Xn_te:
    q = np.round(x / s_in + z_in).clip(-128, 127).astype(np.int8)[None, :]
    interp.set_tensor(i_det['index'], q); interp.invoke()
    pred.append(interp.get_tensor(o_det['index'])[0].argmax())
pred = np.array(pred)
acc_q = (pred == y_te).mean()
print(f"[5] TFLite INT8 — acurácia no teste: {acc_q*100:.1f}%  | tamanho: {len(tflite)/1024:.2f} KB")
print("\nRelatório (modelo INT8):")
print(classification_report(y_te, pred, target_names=NOMES_CLASSES, digits=3))
print("Matriz de confusão (linha = real, coluna = previsto):")
print(confusion_matrix(y_te, pred))

# ── 6. model.h ────────────────────────────────────────────────────────────────
linhas = ['// model.h — gerado por tcc_exportar_tflite_3classes.py', '#pragma once', '#include <stdint.h>', '',
          f'#define N_FEATURES 8', f'#define N_CLASSES 3', '',
          f'const unsigned int modelo_surtos_tamanho = {len(tflite)};',
          'alignas(16) const uint8_t modelo_surtos[] = {']
for i in range(0, len(tflite), 12):
    linhas.append('  ' + ', '.join(f'0x{b:02x}' for b in tflite[i:i+12]) + ',')
linhas += ['};', '',
           '// Normalização min-max (calculada só no treino). Replicar no firmware:',
           '//   x_norm = clamp((x - X_MIN[i]) / (X_MAX[i] - X_MIN[i]), 0, 1)',
           'const float X_MIN[N_FEATURES] = {' + ', '.join(f'{v:.8e}f' for v in x_min) + '};',
           'const float X_MAX[N_FEATURES] = {' + ', '.join(f'{v:.8e}f' for v in x_max) + '};',
           f'const float INPUT_SCALE = {s_in:.10e}f;', f'const int INPUT_ZERO_POINT = {int(z_in)};',
           '// Ordem das features: ' + ', '.join(NOMES_FEAT),
           '// Classes: 0=Normal, 1=Surto Leve, 2=Surto Severo']
open('model.h', 'w').write('\n'.join(linhas))
np.savez('normalizacao_tflite.npz', x_min=x_min, x_max=x_max)
print("\n[6] Arquivos: detector_surtos_int8.tflite | model.h | normalizacao_tflite.npz")
print(f"\nRESUMO  RF {acc_rf*100:.1f}% | MLP float {acc_nn*100:.1f}% | MLP INT8 {acc_q*100:.1f}% | {len(tflite)/1024:.2f} KB")
