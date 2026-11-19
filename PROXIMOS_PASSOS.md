# Próximos passos

## Prioridade 1 — Compreender e validar a base atual

- [x] Executar o notebook do início ao fim sem depender de arquivos antigos do Colab (notebook limpo).
- [ ] Documentar a topologia elétrica equivalente com diagrama e equações.
- [x] Verificar a origem dos parâmetros R/L/C: são os padrão do MATLAB para 735 kV, não de 13,8 kV (ver D-007).
- [ ] Decidir o destino do circuito RLC: remover do texto, trocar por parâmetros de 13,8 kV com fonte, ou acoplar ao dataset.
- [x] Reproduzir a medição da frente e da cauda do impulso (T1 de 1,33 a 1,36 µs, T2 de 51,57 a 51,60 µs, contado a partir da origem virtual).
- [x] Confirmar no código que o dataset não usa as formas de onda simuladas.

## Prioridade 2 — Melhorar os dados e a validação

- [ ] Justificar os critérios de Normal, Surto Leve e Surto Severo (hoje são faixas de amplitude operacionais).
- [x] **Teste de robustez** com 7 cenários e 10 sementes (D-010).
- [x] Verificar a resolução em frequência e a validade das características (D-005).
- [x] Registrar matriz de confusão, precisão, recall e F1 por classe.
- [x] Teste separado por cenário (robustez). Limitação: mesma família de geradores.

## Prioridade 3 — Comparar modelos

- [x] Baseline de Random Forest reproduzível (sementes fixas).
- [x] Rede neural pequena avaliada (float32 98,7%, INT8 98,3%).
- [x] Comparar desempenho e tamanho (RF 99,0% contra INT8 98,3% e 3,47 KB).
- [ ] Salvar cada experimento em `experimentos/` com parâmetros e sementes.

## Prioridade 4 — Preparar TinyML

- [x] Exportar para TFLite INT8 e gerar `firmware/model.h`.
- [x] Verificar que o `model.h` equivale ao `.tflite` (100% de concordância).
- [ ] Escrever o firmware do ESP32-S3 (normalizar, quantizar com `roundf()`, classificar).
- [ ] Redefinir as características para a taxa real do ADC (ESP-IDF v4.4.6: 611 Hz a 83333 Hz no ADC contínuo).
- [ ] Definir a cadeia de aquisição segura (sensor, condicionamento, isolamento).
- [ ] Medir tempo de inferência e RAM no chip, se houver a placa.

## Prioridade 5 — Texto do TCC

- [x] Resumo, palavras-chave, introdução e objetivos atualizados.
- [ ] Referencial teórico (depende da leitura das fontes).
- [ ] Investigar por que o cenário D (impulso nas bordas da janela) é o pior (opcional).
- [ ] Abstract e keywords em inglês.
- [ ] Slides, roteiro de fala e perguntas da banca.
- [x] Materiais e métodos.
- [x] Resultados e discussão.
- [x] Conclusão e trabalhos futuros.

## Registro de progresso

Atualize as caixas e registre datas, arquivos e resultados em `experimentos/`. Só marque uma tarefa como concluída com evidência reproduzível.
