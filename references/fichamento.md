# Fichamento de referências — TP1 (G2, RSNA 2020 Pulmonary Embolism Detection)

> **Nota de integridade acadêmica.** Esta lista foi montada originalmente com apoio de
> IA generativa (Claude), a partir do conhecimento do modelo, sem acesso à internet.
> Posteriormente, todas as 11 entradas foram **verificadas via busca web** (autor,
> título exato, veículo, volume/número, páginas e DOI, quando disponível) — ver
> `article/references.bib` para os DOIs confirmados. Nessa verificação, **a referência
> nº 11 (Huang et al., PENet) tinha o título errado** na versão original (dizia
> "a Scan-level Deep-learning Model for Automatic Diagnosis..."; o título correto é "a
> Scalable Deep-learning Model for Automated Diagnosis... Using Volumetric CT
> Imaging") e foi corrigida, junto com a lista completa de autores. As demais 10
> entradas foram confirmadas como corretas (autor, veículo, volume, páginas batendo
> com a fonte original). Ainda assim, o grupo deve dar uma conferida final antes da
> entrega — verificação por IA, mesmo com busca web, não substitui a responsabilidade
> do grupo (enunciado §6).

---

## 1. Colak, E., Kitamura, F. C., Hobbs, S. B. et al. (2021)
**"The RSNA Pulmonary Embolism CT Dataset."** *Radiology: Artificial Intelligence.*

**O que é:** o artigo de descrição oficial do desafio atribuído ao grupo (RSNA STR
Pulmonary Embolism Detection, 2020). Descreve como os ~12.000+ exames de angio-TC de
tórax foram coletados (múltiplas instituições/continentes), como os rótulos foram
anotados (nível de corte **e** de exame, com regras de consistência hierárquica:
`negative_exam_for_pe`, `rv_lv_ratio_gte_1`, `indeterminate`, lateralidade,
agudo/crônico etc.) e as métricas oficiais de avaliação da competição Kaggle.

**Relevância direta:** é a fonte primária da formulação da tarefa (§4 da Metodologia)
e da justificativa da unidade de análise (exame vs. corte). Também é a referência
para os números oficiais de prevalência de EP na base, usados para dimensionar a
amostra estratificada do baseline.

**Leitura crítica:** por ser o artigo dos próprios organizadores, não avalia métodos
de detecção — é puramente descritivo do dado. Não dispensa a leitura do dicionário
de dados oficial do Kaggle (mencionado no enunciado §12), que traz o esquema exato de
colunas do `train.csv`, necessário para construir o manifesto e os rótulos de exame.

---

## 2. Haralick, R. M., Shanmugam, K., Dinstein, I. (1973)
**"Textural Features for Image Classification."** *IEEE Transactions on Systems, Man,
and Cybernetics*, SMC-3(6), 610–621.

**O que é:** artigo seminal que introduz a matriz de co-ocorrência de níveis de cinza
(GLCM) e um conjunto de 14 descritores estatísticos derivados dela (contraste,
homogeneidade, energia, correlação etc.), usados até hoje como baseline de textura em
imagem médica.

**Relevância direta:** fundamenta `src/features/texture.py::extract_glcm_haralick`,
uma das famílias de descritores obrigatórias do baseline (categoria "Textura",
enunciado §4.2).

**Leitura crítica:** os descritores de Haralick não são naturalmente invariantes à
rotação; nosso pipeline mitiga isso calculando a GLCM em 4 ângulos e reportando
média/desvio — uma escolha metodológica citável e defensável, mas que ainda assim
não é equivalente a um descritor formalmente rotação-invariante (ponto explicitado
na Metodologia do artigo).

---

## 3. Ojala, T., Pietikäinen, M., Mäenpää, T. (2002)
**"Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local
Binary Patterns."** *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
24(7), 971–987.

**O que é:** propõe o LBP uniforme (`LBP^{u2}_{P,R}`), invariante à rotação e robusto
a transformações monotônicas de intensidade (ganho/offset de brilho) — propriedade
particularmente relevante quando diferentes protocolos de janelamento (window
center/width) são usados entre instituições, como é o caso de uma base multicêntrica
como a do RSNA.

**Relevância direta:** fundamenta `src/features/texture.py::extract_lbp`.

**Leitura crítica:** a invariância do LBP é a transformações *monotônicas*; o
janelamento em HU aplicado por nós é monotônico dentro da faixa não saturada, mas
introduz saturação (clipping) nas extremidades — outra decisão explicitamente
discutida e citada, não assumida como "óbvia" (exigência do enunciado sobre
embasamento de decisões metodológicas).

---

## 4. Dalal, N., Triggs, B. (2005)
**"Histograms of Oriented Gradients for Human Detection."** *Proceedings of the 2005
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 886–893.

**O que é:** introduz o descritor HOG — histogramas de gradiente orientados,
normalizados em blocos sobrepostos — originalmente para detecção de pedestres, mas
adotado amplamente em CAD médico por capturar bem estruturas alongadas/tubulares.

**Relevância direta:** fundamenta `src/features/gradient.py`, família "Gradiente e
bordas" do baseline.

**Leitura crítica:** HOG foi desenhado para imagens naturais em escala de cinza de 8
bits; aplicá-lo a HU janelado exige a normalização prévia para [0,1] que já fazemos no
pré-processamento — sem ela, os gradientes calculados ficariam em escala arbitrária e
não comparável entre exames com diferentes RescaleSlope/Intercept.

---

## 5. van Griethuysen, J. J. M., Fedorov, A., Parmar, C. et al. (2017)
**"Computational Radiomics System to Decode the Radiographic Phenotype."** *Cancer
Research*, 77(21), e104–e107.

**O que é:** artigo de referência da biblioteca **PyRadiomics**, implementação de
código aberto compatível com o padrão IBSI (Image Biomarker Standardisation
Initiative), hoje o padrão de fato para extração radiômica reprodutível em pesquisa
clínica.

**Relevância direta:** fundamenta `src/features/radiomics_features.py`, família
"Radiômica" — explicitamente sugerida pelo enunciado (§4.2) como "bastante adequada
ao domínio médico e bem sustentada na literatura".

**Leitura crítica:** PyRadiomics pressupõe uma **máscara/ROI válida**; no baseline,
por não termos segmentação anatômica supervisionada disponível, usamos um limiar
simples de intensidade como proxy de ROI — limitação relevante para a análise de
erro e discutida na Conclusão (trabalho futuro: segmentação de vasos/pulmão).

---

## 6. Bi, J., Liang, J. (2007)
**"Multiple instance learning of pulmonary embolism detection with geodesic distance
along vascular structure."** *Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR)*.

**O que é:** um dos trabalhos clássicos (pré-aprendizado profundo) de CAD para EP em
angio-TC, usando características hand-crafted extraídas ao longo da estrutura
vascular segmentada e classificadores baseados em aprendizado multi-instância — a
mesma família de abordagem (características manuais + classificador clássico) exigida
como restrição central deste TP.

**Relevância direta:** é a peça central dos "Trabalhos Relacionados" do artigo — mostra
que a abordagem clássica não é apenas permitida pelo enunciado, mas tem precedente
direto na literatura específica de EP, útil para justificar as escolhas de descritores.

**Leitura crítica:** os autores dependiam de uma segmentação vascular prévia (mais
sofisticada que o limiar simples do nosso baseline) — diferença que deve ser
explicitada como limitação ao comparar resultados na Conclusão.

---

## 7. Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002)
**"SMOTE: Synthetic Minority Over-sampling Technique."** *Journal of Artificial
Intelligence Research*, 16, 321–357.

**O que é:** técnica de sobre-amostragem sintética da classe minoritária por
interpolação entre vizinhos mais próximos no espaço de características.

**Relevância direta:** fundamenta o tratamento de desbalanceamento opcional em
`src/models/classical.py` (classe `SafeSMOTE`) — a base de EP costuma ter
desbalanceamento relevante entre exames positivos/negativos (ver Colak et al., 2021).

**Leitura crítica:** SMOTE pode gerar exemplos sintéticos implausíveis quando aplicado
sobre características de alta dimensionalidade e poucos exemplos minoritários — por
isso o artigo deve reportar honestamente os "efeitos colaterais" (enunciado §4.3),
comparando com pesos de classe (`class_weight='balanced'`) como alternativa.

---

## 8. Breiman, L. (2001)
**"Random Forests."** *Machine Learning*, 45(1), 5–32.

**O que é:** artigo fundador do Random Forest — ensemble de árvores de decisão
treinadas em amostras bootstrap com seleção aleatória de características por split.

**Relevância direta:** fundamenta o modelo `random_forest` em
`src/models/classical.py`; também permite importância de características por
permutação (mencionada no enunciado §4.3), útil para responder "quais características
carregam sinal" na análise de erro.

**Leitura crítica:** a importância de características *nativa* do Random Forest
(baseada em impureza) é enviesada a favor de variáveis contínuas de alta
cardinalidade — por isso preferimos, quando possível, importância por permutação
(`sklearn.inspection.permutation_importance`) na análise final.

---

## 9. Chen, T., Guestrin, C. (2016)
**"XGBoost: A Scalable Tree Boosting System."** *Proceedings of the 22nd ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining (KDD)*, 785–794.

**O que é:** algoritmo de gradient boosting regularizado, amplamente adotado como
baseline forte em dados tabulares/vetores de características.

**Relevância direta:** fundamenta o modelo `xgboost` em `src/models/classical.py`,
satisfazendo o requisito de "gradient boosting (XGBoost/LightGBM)" do enunciado §4.3.

**Leitura crítica:** XGBoost tem muitos hiperparâmetros sensíveis (profundidade,
learning rate, regularização); o ajuste deve ocorrer em validação cruzada aninhada
(nunca no conjunto de teste), como implementado em
`src/evaluation/protocol.py::run_cross_validation`.

---

## 10. Cortes, C., Vapnik, V. (1995)
**"Support-Vector Networks."** *Machine Learning*, 20(3), 273–297.

**O que é:** artigo fundador das Support Vector Machines (SVM), com a formulação de
margem máxima e o "kernel trick" para fronteiras de decisão não lineares (ex.: RBF).

**Relevância direta:** fundamenta `svm_linear` e `svm_rbf` em
`src/models/classical.py`, satisfazendo o requisito "SVM (linear e RBF)" do
enunciado §4.3.

**Leitura crítica:** SVM com kernel RBF é sensível à escala das características —
reforça a necessidade de padronização (`StandardScaler`) dentro do mesmo Pipeline
ajustado por dobra, já implementada em `_with_preprocessing`.

---

## 11. Huang, S.-C., Kothari, T., Banerjee, I. et al. (2020)
**"PENet — a scalable deep-learning model for automated diagnosis of pulmonary
embolism using volumetric CT imaging."** *npj Digital Medicine*, 3, 61.
DOI: 10.1038/s41746-020-0266-y. (Título corrigido após verificação via busca web —
a versão original desta ficha tinha "scan-level"/"automatic diagnosis", incorreto.)

**O que é:** modelo de aprendizado profundo (CNN 3D) para detecção de EP em nível de
exame, com um dos conjuntos de dados de referência anteriores ao do RSNA.

**Relevância direta:** usado nos "Trabalhos Relacionados" como contraponto — o
propósito deste TP não é competir com este tipo de abordagem, mas sim estabelecer o
piso de desempenho (baseline) contra o qual uma futura réplica de algo como o PENet
deverá ser comparada (ver Objetivo, §1 do enunciado).

**Leitura crítica:** por usar redes profundas de ponta a ponta, este trabalho está
**fora do escopo permitido** deste TP (restrição central do enunciado) — citado apenas
como referência de literatura relacionada, nunca como método a ser reproduzido aqui.

---

## Observação sobre o mínimo de referências

O enunciado exige **mínimo de 10 referências no artigo**, **pelo menos 6 de periódicos
ou conferências revisados por pares** — as 11 entradas acima já superam ambos os
limiares (10 são de periódico/conferência revisados por pares; a exceção parcial é a
nº 1, que é o artigo descritivo do próprio dataset, também publicado em periódico
revisado por pares). Referências adicionais específicas do desafio (ex.: trabalhos
que usem features hand-crafted em angio-TC pulmonar, achados sobre RV/LV ratio,
literatura sobre `StratifiedGroupKFold`/vazamento de dados em imagem médica) devem ser
incorporadas pelo grupo ao longo da Semana 1–2, conforme a leitura aprofundar.
