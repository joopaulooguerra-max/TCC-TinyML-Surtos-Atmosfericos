# Próximos passos

## Prioridade 1 — Compreender e validar a base atual

- [ ] Executar todas as células do notebook do início ao fim e confirmar que os arquivos são gerados sem depender de artefatos antigos do Colab.
- [ ] Documentar a topologia elétrica equivalente com um diagrama e equações coerentes.
- [ ] Verificar unidades, parâmetros R/L/C e a interpretação da tensão de 13,8 kV.
- [ ] Reproduzir a medição da frente e da cauda do impulso; registrar o método de cálculo e as referências usadas.
- [ ] Confirmar, no código, se os exemplos de treinamento usam as formas de onda simuladas ou se são gerados por funções independentes.

## Prioridade 2 — Melhorar os dados e a validação

- [ ] Definir critérios operacionais e justificáveis para Normal, Surto Leve e Surto Severo.
- [ ] Incluir variações justificadas de parâmetros, amplitude, fase, ruído e instante do evento.
- [ ] Verificar a resolução em frequência e a validade das oito características atuais.
- [ ] Criar um teste separado por cenário/configuração, não apenas por amostras aleatórias da mesma distribuição.
- [ ] Registrar matriz de confusão, precisão, recall, F1-score e resultados por cenário.

## Prioridade 3 — Comparar modelos

- [ ] Estabelecer um baseline reproduzível de Random Forest.
- [ ] Treinar e avaliar uma MLP, se continuar alinhada ao escopo do TCC.
- [ ] Comparar desempenho, estabilidade, custo computacional e complexidade.
- [ ] Salvar versões dos experimentos, parâmetros e sementes aleatórias.

## Prioridade 4 — Preparar TinyML

- [ ] Definir o sinal que seria medido e a cadeia de sensor/condicionamento/isolamento de forma segura.
- [ ] Medir tempo de inferência, memória e tamanho do modelo.
- [ ] Avaliar exportação/quantização INT8 somente após selecionar um modelo estável.
- [ ] Implementar e testar no ESP32-S3 com sinais de teste controlados e seguros.
- [ ] Documentar limitações; não usar o protótipo como proteção elétrica operacional.

## Registro de progresso

Atualize as caixas de seleção e registre datas, arquivos alterados e resultados em `experimentos/`. Não marque uma tarefa como concluída sem uma evidência reproduzível.
