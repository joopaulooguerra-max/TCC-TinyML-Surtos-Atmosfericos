# Contexto do projeto

## 1. Identificação

- **Título de trabalho:** Sistema Embarcado Baseado em TinyML para Classificação de Surtos Atmosféricos em Redes de Distribuição Elétrica.
- **Área:** Engenharia da Computação, sistemas embarcados, processamento de sinais e aprendizado de máquina.
- **Plataforma embarcada pretendida:** ESP32-S3.
- **Ambiente de desenvolvimento:** Python / Google Colab.
- **Estado:** classificação em simulação concluída (dados sintéticos), modelo exportado para TFLite INT8 e inferência validada em Python. Firmware, validação em hardware e com sinais reais pendentes.

## 2. Objetivo atual

Investigar, em simulação, a viabilidade de classificar sinais em três classes (operação normal, surto leve e surto severo) com características dos domínios do tempo e da frequência, usando um modelo pequeno o bastante para um microcontrolador.

Descrever sempre como **classificação em cenários simulados**. Não afirmar que o sistema prevê surtos, protege uma rede ou está pronto para uso operacional.

## 3. Pipeline atual

Notebook: `notebooks/TCC_Surtos_Notebook_Limpo.ipynb` (Executar tudo). Etapas:

1. **Simulação (`simulacao/tcc_simulacao_v2.py`):** circuito RLC série equivalente simplificado, passo de 100 ns, janela de 200 µs. Serve para visualizar o impulso e verificar T1/T2. **Não alimenta o dataset.**
2. **Dataset (`simulacao/tcc_augmentation_3classes.py`):** 1500 sinais (500 por classe, 2001 amostras). Cada sinal é gerado por função própria: senoide de 60 Hz (amplitude e fase aleatórias) + ruído gaussiano + impulso de dupla exponencial em posição aleatória, em por unidade (1,0 = nominal).
3. **Pipeline (`modelos/tcc_pipeline_3classes_v2.py`):** 8 características e Random Forest, com validação cruzada k = 5.
4. **Exportação (`modelos/tcc_exportar_tflite_3classes.py`):** rede neural 8-16-8-3 com softmax, divisão 60/20/20, normalização calculada só no treino, TFLite INT8 e `firmware/model.h`.
5. **Validação (`modelos/tcc_validar_inferencia_int8.py`):** compara `model.h` e inferência refeita como o C faria com o modelo `.tflite`.

### Características (versão atual)

RMS, pico, fator de crista, desvio padrão, proporção de energia acima de 100 kHz, centroide espectral, frequência dominante acima de 10 kHz e entropia espectral. A magnitude em 60 Hz foi removida (inválida: a resolução da FFT é 5 kHz por bin) e a energia foi removida (redundante com o RMS).

## 4. Resultados (dados sintéticos)

| Avaliação | Resultado |
|---|---|
| Random Forest, validação cruzada k = 5 | 97,8% ± 0,62% |
| Random Forest, teste do pipeline (80/20) | 98%, F1 de 1,00 / 0,96 / 0,97 por classe |
| Random Forest, teste da exportação (60/20/20) | 99,0% |
| Rede neural float32 / INT8 | 98,7% / 98,3% |
| Tamanho do modelo INT8 | 3,47 KB |
| `model.h` x `.tflite` | bytes idênticos, 100% de concordância na inferência |

Estes números valem para a distribuição sintética usada. Não são evidência de desempenho em dados reais.

## 5. Riscos e estado

1. **Simulação x dados — resolvido (verificado no código).** O dataset não deriva da simulação RLC. Descrever assim no TCC.
2. **Modelo elétrico — resolvido.** É RLC série simplificado, não uma linha PI. Os valores de R, L e C são os padrão do MATLAB/Simscape para uma linha de 735 kV e não representam 13,8 kV. Foram atribuídos por engano a Stevenson em versões anteriores (corrigido).
3. **Impulso 1,2/50 µs — verificado.** `alfa` e `beta` foram ajustados numericamente. A medição dá T1 de 1,33 µs e T2 de 51,40 µs com passo de 100 ns (1,36 µs e 51,3 µs com passo de 1 ns), dentro das tolerâncias da IEC 60060-1 (±30% e ±20%). T2 foi medido a partir de t = 0, sem a origem virtual da norma.
4. **Classes — aberto.** As faixas de amplitude (1,5 a 3× e 3 a 6×) são definição operacional deste trabalho. Elas se sobrepõem nas bordas, o que explica os erros entre leve e severo.
5. **Características espectrais — resolvido.** Limiares ajustados para a janela de 200 µs. As características de frequência têm pouca importância.
6. **Validação por cenário — aberto.** O teste atual usa amostras aleatórias do mesmo gerador. Falta um teste de robustez (ruído, amplitude e instante em faixas não vistas no treino).
7. **Aquisição embarcada — aberto.** A simulação usa 10 MHz e o ADC do ESP32-S3 chega a cerca de 100 mil amostras por segundo. As características de 100 kHz e 10 kHz precisariam ser redefinidas. Média tensão nunca vai direto ao ADC.
8. **TinyML — parcialmente concluído.** Exportação INT8 e validação em Python concluídas. Faltam firmware, tempo de inferência e memória medidos no chip.

## 6. Escopo e limitações

O trabalho é válido em simulação se os pressupostos estiverem descritos e as conclusões se limitarem ao que foi testado. A falta de dados reais e de validação em hardware é limitação declarada e trabalho futuro. O protótipo não é um equipamento de proteção validado e não deve ser conectado a uma rede real.

## 7. Próxima prioridade

Escrever o firmware do ESP32-S3 (usando `roundf()` e saturação na quantização), rodar o teste de robustez dos dados e definir a cadeia de aquisição segura.

## 8. Como atualizar este arquivo

Atualize quando mudar objetivo, metodologia, parâmetros, resultados ou limitações. Registre decisões em `DECISOES.md` e experimentos em `experimentos/`.
