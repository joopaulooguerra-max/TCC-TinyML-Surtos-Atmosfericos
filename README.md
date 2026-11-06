# TCC — Classificação de Surtos Atmosféricos com TinyML

**Tema:** Sistema Embarcado Baseado em TinyML para Classificação de Sinais de Surtos Atmosféricos em Redes de Distribuição Elétrica

**Autores:** Bruno Morais, Eytor Bezerra, João Paulo  · **Orientador:** João Ferreira

> **Escopo:** estudo de viabilidade feito **inteiramente em simulação**. Os dados são sintéticos e não há validação com medições reais nem em hardware.

## O que o projeto faz

Classifica uma janela de sinal de tensão em três classes e exporta o modelo para o formato que roda em microcontrolador (ESP32-S3).

| Classe | Nome | Definição usada neste trabalho |
|:--:|---|---|
| 0 | Normal | Senoide de 60 Hz com ruído |
| 1 | Surto leve | Impulso com pico de 1,5× a 3× a amplitude nominal |
| 2 | Surto severo | Impulso com pico de 3× a 6× a amplitude nominal |

As faixas de amplitude são uma **definição operacional** deste trabalho, não limites universais. Por isso as classes leve e severo se sobrepõem nas bordas.

## Estrutura

```
simulacao/   tcc_simulacao_v2.py, tcc_augmentation_3classes.py
modelos/     tcc_pipeline_3classes_v2.py, tcc_exportar_tflite_3classes.py,
             tcc_validar_inferencia_int8.py, detector_surtos_int8.tflite, normalizacao_3classes_v2.npz
firmware/    model.h   (modelo INT8 + constantes de normalização)
dados/       (gerado pelos scripts, não versionado)
resultados/  resultados_v2.png
```

## Como os dados são gerados

- **Dataset (`tcc_augmentation_3classes.py`):** cada sinal = senoide de 60 Hz (amplitude e fase aleatórias) + ruído gaussiano + (nas classes de surto) um impulso de dupla exponencial em posição aleatória. Amplitudes em por unidade (1,0 = nominal). 1500 sinais, 500 por classe, janela de 200 µs a 10 MHz (2001 amostras).
- **Impulso 1,2/50 µs:** `v(t) ∝ e^(−αt) − e^(−βt)` com α = 14286 s⁻¹ e β = 2,16×10⁶ s⁻¹. Esses valores foram **ajustados numericamente** até a onda atender às tolerâncias da IEC 60060-1. Medido: frente T1 = 1,36 µs (tolerância ±30%) e cauda T2 = 51,3 µs (±20%).
- **Simulação RLC (`tcc_simulacao_v2.py`):** circuito RLC simplificado, usado para visualizar o impulso e verificar T1/T2. **Não alimenta o dataset.** Os valores de R, L e C são os parâmetros padrão do MATLAB/Simscape para uma linha de **735 kV**, e não representam uma linha de 13,8 kV.

## Características extraídas (8)

RMS, pico, fator de crista, desvio padrão, proporção de energia acima de 100 kHz, centroide espectral, frequência dominante acima de 10 kHz e entropia espectral.

Observações: uma versão anterior usava a magnitude em 60 Hz, que era inválida (a resolução da FFT na janela de 200 µs é de 5 kHz por bin) e foi removida. As características de frequência contribuem pouco, porque o impulso tem componentes acima do que a taxa de 10 MHz resolve bem.

## Resultados (dados sintéticos)

| Avaliação | Resultado |
|---|---|
| Random Forest, validação cruzada k = 5 | **97,8% ± 0,62%** (folds: 98,3; 97,3; 98,7; 97,0; 97,7) |
| Random Forest, conjunto de teste (60/20/20) | 99,0% |
| Rede neural float32, teste | 98,7% |
| **Rede neural INT8 (TFLite), teste** | **98,3%** |
| Tamanho do modelo INT8 | 3,47 KB (3552 bytes) |

**Validação da inferência (`tcc_validar_inferencia_int8.py`):** os bytes do `model.h` são idênticos ao `.tflite`, e a inferência refeita como o C faria (normalização em float32, `roundf`, saturação em int8) concorda 100% com o interpretador TFLite. Truncar em vez de arredondar na quantização muda previsões, então o firmware deve usar `roundf()`.

## Limitações

- Dados sintéticos gerados pelo mesmo modelo usado para definir as classes: a acurácia mede a separação dentro desse modelo, não o desempenho em redes reais.
- O impulso 1,2/50 µs é um padrão de ensaio e não cobre a variedade de raios reais.
- A simulação usa 10 MHz. O ADC do ESP32-S3 é bem mais lento (cerca de 100 mil amostras/s), então as características de 100 kHz e 10 kHz precisariam ser redefinidas para aquisição real.
- A validação confirma o modelo e a aritmética de quantização. Não valida a extração de características em C, o ADC nem o tempo de inferência no chip.

## Como executar

Use o notebook `TCC_Surtos_Notebook_Limpo.ipynb` no Google Colab (Executar tudo). Ordem dos scripts: simulação → augmentation → pipeline → exportação TFLite → validação.

## Próximos passos

| Etapa | Status |
|---|---|
| Dataset, pipeline, K-Fold | Concluído |
| Exportação TFLite INT8 e `model.h` | Concluído |
| Validação da inferência em Python | Concluído |
| Firmware ESP32-S3 (C++) | Pendente |
| Validação em hardware e com sinais reais | Trabalho futuro |

## Referências

- INTERNATIONAL ELECTROTECHNICAL COMMISSION. IEC 60060-1: High-voltage test techniques, Part 1.
- ELAT/INPE. Grupo de Eletricidade Atmosférica. http://www.inpe.br/webelat/
- WARDEN, P.; SITUNAYAKE, D. TinyML. O'Reilly Media.
- ESPRESSIF SYSTEMS. ESP32-S3 Series Datasheet.
- MATHWORKS. PI Section Line (parâmetros padrão da linha de 735 kV). https://www.mathworks.com/help/sps/powersys/ref/pisectionline.html
