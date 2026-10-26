# Contexto do projeto

## 1. Identificação

- **Título de trabalho:** Sistema Embarcado Baseado em TinyML para Classificação de Surtos Atmosféricos em Redes de Distribuição Elétrica.
- **Área:** Engenharia da Computação, sistemas embarcados, processamento de sinais e aprendizado de máquina.
- **Plataforma embarcada pretendida:** ESP32-S3.
- **Ambiente atual de desenvolvimento:** Python / Google Colab.
- **Estado:** protótipo de simulação e classificação com dados sintéticos; validação física e embarcada ainda pendente.

## 2. Objetivo atual

Investigar a viabilidade de classificar sinais sintéticos/simulados em três classes — operação normal, surto leve e surto severo — utilizando características no domínio do tempo e da frequência e algoritmos de aprendizado de máquina. Uma etapa futura prevista é avaliar modelos compactos para execução no ESP32-S3.

O objetivo deve ser descrito como **classificação em cenários simulados** enquanto não houver avaliação com sinais reais ou com uma referência independente validada. Evitar afirmar que o sistema já prevê surtos, protege uma rede elétrica ou está pronto para uso operacional.

## 3. Pipeline existente

O notebook `notebooks/ClassificadorTCC.ipynb` contém três células de código principais:

1. **Simulação:** modelo elétrico equivalente simplificado de uma linha de distribuição de 13,8 kV / 10 km, com passo temporal de 100 ns e janela de 200 µs. A saída registrada informa um impulso com frente de aproximadamente 1,33 µs e cauda de 51,40 µs, dentro das tolerâncias que o próprio notebook atribui à IEC 60060-1. Essa verificação deve ser reproduzida e revisada antes de fazer uma afirmação normativa definitiva.
2. **Geração de exemplos:** 1.500 janelas sintéticas, com 500 exemplos por classe e 2.001 amostras por janela. As variações descritas incluem amplitude, fase, ruído, posição temporal do evento e amplitude do pico.
3. **Classificação:** extração de oito características e treinamento de Random Forest, com divisão treino/teste e validação cruzada de cinco folds.

### Características usadas no notebook atual

- RMS;
- valor de pico;
- fator de crista;
- energia;
- magnitude próxima de 60 Hz;
- energia de alta frequência;
- frequência dominante;
- entropia espectral.

A definição exata de cada característica e os detalhes da extração devem ser confirmados diretamente no código antes de qualquer modificação.

## 4. Resultados preliminares do notebook

O resultado salvo no notebook apresenta:

- **Acurácia no conjunto de teste:** 98% (300 exemplos no teste, 100 por classe).
- **F1-score por classe:** 1,00 para Normal; 0,97 para Surto Leve; 0,97 para Surto Severo.
- **Validação cruzada de cinco folds:** média de 97,73%, desvio-padrão aproximado de 0,65 ponto percentual.

Esses resultados são preliminares e se referem à distribuição sintética usada no experimento. Eles não são evidência de desempenho em dados reais. A avaliação deve ser repetida após revisar a geração dos sinais, as características e a estratégia de divisão dos dados.

## 5. Riscos técnicos identificados para revisão

1. **Coerência entre simulação e dados:** verificar se os exemplos de treinamento derivam de forma rastreável das saídas da simulação elétrica ou se são gerados por funções artificiais independentes. O notebook precisa deixar explícita essa relação.
2. **Modelo elétrico:** documentar o circuito equivalente e verificar a topologia, unidades, parâmetros e interpretação da capacitância. Não descrever o modelo simplificado como uma linha PI completa se não representar essa topologia.
3. **Forma de onda de impulso:** reproduzir o cálculo de frente e cauda e registrar como os tempos são medidos. Não afirmar conformidade normativa sem evidência e referência adequadas.
4. **Classes:** justificar os critérios usados para separar surto leve e severo. Os rótulos atuais são escolhas operacionais do experimento, não níveis universais para todas as redes.
5. **Características espectrais:** verificar a resolução em frequência e se as características em torno de 60 Hz e de alta frequência correspondem ao que a janela temporal consegue resolver.
6. **Validação:** separar cenários completos ou configurações de simulação entre treino e teste, para reduzir o risco de o teste conter exemplos quase derivados das mesmas condições do treino.
7. **Aquisição embarcada:** especificar qual grandeza seria medida, o sensor/condicionamento, isolamento, taxa de amostragem, faixa dinâmica e segurança. O ADC do ESP32-S3 não deve ser ligado diretamente a média tensão.
8. **TinyML:** a exportação para TFLite/INT8 e a execução no ESP32-S3 são etapas futuras; não registrar como concluídas sem artefatos e testes verificáveis.

## 6. Escopo e limitações

O trabalho pode ser academicamente válido usando simulação, desde que a metodologia seja coerente, os pressupostos estejam descritos, os experimentos sejam reproduzíveis e as conclusões se limitem ao que foi testado. A ausência de dados reais deve ser registrada como limitação e oportunidade para trabalho futuro.

O protótipo não é um equipamento de proteção elétrica validado e não deve ser conectado a uma rede real para atuação.

## 7. Próxima prioridade

Antes de otimizar métricas ou converter modelos para microcontrolador, revisar o vínculo entre a simulação elétrica e a geração de dados e validar as características do sinal. Em seguida, definir um protocolo de teste com cenários inteiros reservados e documentar cada execução.

## 8. Como atualizar este arquivo

Atualize este documento quando houver uma mudança importante de objetivo, metodologia, parâmetros, resultados ou limitações. Registre decisões com data em `DECISOES.md` e registre cada experimento em `experimentos/`.
