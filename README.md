# TP1 — Baseline de Projeto de Pesquisa (Grupo G2)

**Disciplina:** Tópicos Especiais em Sistemas de Informação
**Desafio RSNA atribuído:** 2020 — *Pulmonary Embolism Detection*
**Integrantes:** Rawenne da Silva Leite · Wheverson de Abreu Lima · João Victor do Nascimento Silva

Este repositório contém o baseline clássico (sem redes neurais profundas) construído
para o desafio RSNA STR Pulmonary Embolism Detection: leitura/pré-processamento de
DICOM, extração de cinco famílias de características hand-crafted, sete modelos
clássicos (incluindo o baseline trivial obrigatório) e um protocolo experimental
sem vazamento de dados (partição por paciente).

## ⚠️ Leia antes de tudo: sobre os dados usados neste repositório

O ambiente usado para montar este repositório **não tinha acesso ao Kaggle**
(rede corporativa bloqueava `kaggle.com`). Por isso, todo o código foi validado
de ponta a ponta com um conjunto de **exames DICOM sintéticos** (elipsoides
gerados por código, sem significado clínico — ver `src/data/synthetic.py`),
não com os dados reais do desafio.

**O que isso significa na prática:**
- Todo o pipeline (leitura DICOM → pré-processamento → extração de características
  → modelagem → avaliação) está implementado, testado e funcional.
- As tabelas/figuras já geradas em `outputs/` e os números de exemplo em
  `article/main.tex` (Seção 4, Resultados) vêm dos dados sintéticos — **não têm
  valor científico** e precisam ser regenerados com os dados reais antes da entrega.
- O grupo precisa apenas: (1) baixar a amostra real (passo a passo abaixo),
  (2) gerar um manifesto real no mesmo formato do sintético, (3) rodar
  `scripts/extract_features.py` e `scripts/run_experiment.py` de novo.

## 1. Configuração do ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `pyradiomics` tem uma dependência de build que exige `numpy` já instalado. Se
> `pip install -r requirements.txt` falhar nessa etapa específica, rode:
> ```bash
> pip install -r <(grep -v pyradiomics requirements.txt)
> pip install pyradiomics==3.0.1 --no-build-isolation
> ```
> (A versão `3.1.0` do PyPI tem metadata quebrada — ver comentário no
> `requirements.txt`. `3.0.1` foi a versão testada neste repositório.)

Teste rápido de que tudo está funcionando (usa dados sintéticos gerados em
memória, não precisa de nenhum dado externo):

```bash
python -m pytest tests/ -v
```

## 2. Obtendo os dados reais do desafio

1. Aceite os termos de uso da competição em
   https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/rules
   (usar a conta pessoal de cada integrante — cadastro é responsabilidade do grupo,
   conforme §2 do enunciado).
2. Configure a API do Kaggle (`~/.kaggle/kaggle.json`) e baixe os dados:
   ```bash
   pip install kaggle
   kaggle competitions download -c rsna-str-pulmonary-embolism-detection -p data/raw/
   unzip data/raw/rsna-str-pulmonary-embolism-detection.zip -d data/raw/
   ```
   **Atenção:** o conjunto completo tem centenas de GB. Não é necessário (nem
   esperado) baixar tudo — ver amostragem abaixo.
3. Construa uma **amostra estratificada** (por classe e por paciente, semente
   fixa) a partir de `data/raw/train.csv` e de um subconjunto de
   `StudyInstanceUID`s. Documente aqui o critério de inclusão usado (ex.: N
   pacientes positivos + N negativos, semente=42) assim que definido — este
   README deve ser atualizado com o critério real antes da entrega, conforme
   exigência de reprodutibilidade do enunciado (§4.5).
4. Gere um manifesto no mesmo formato de `data/sample/synthetic_manifest.json`
   (lista de objetos com `patient_id`, `study_instance_uid`, `exam_dir`,
   `pe_present_on_exam`), apontando `exam_dir` para as pastas com os `.dcm`
   reais baixados.

## 3. Rodando o pipeline completo

```bash
# 0) (opcional) gerar dados sintéticos de exemplo, só para validar o pipeline
python scripts/generate_synthetic_sample.py

# 1) extrair características (troque --manifest/--data-root para os dados reais)
python scripts/extract_features.py \
    --manifest data/sample/synthetic_manifest.json \
    --data-root data/sample/synthetic \
    --out outputs/tables/features.parquet

# 2) treinar e avaliar todos os modelos (baseline trivial + 6 clássicos)
python scripts/run_experiment.py --features outputs/tables/features.parquet --n-splits 5 --tune
```

Saídas geradas:
- `outputs/tables/results_summary.csv` — tabela comparativa descritor×modelo
  (média ± desvio entre dobras) — é a fonte da Tabela 1 do artigo.
- `outputs/figures/roc_<modelo>.png`, `pr_<modelo>.png`,
  `confusion_matrix_<modelo>.png` — para o melhor modelo (maior AUC-ROC médio).
- `outputs/models/<modelo>.joblib` — modelos finais serializados.

Depois de rodar sobre os dados reais, **copie os números de
`outputs/tables/results_summary.csv` para a Tabela 1 de `article/main.tex`** e
insira as figuras geradas, removendo a "Nota metodológica" da Seção 4.

## 4. EDA

`notebooks/01_eda.ipynb` — distribuição de classes, metadados DICOM, exemplos
visuais por classe. Por padrão lê o manifesto sintético; troque
`MANIFEST_PATH`/`DATA_ROOT` no notebook para os dados reais assim que
disponíveis.

## 5. Estrutura do repositório

```
src/                   # código de biblioteca (importável, testado)
  data/                 # leitura DICOM (dicom_io.py), dados sintéticos (synthetic.py)
  preprocessing/         # janelamento, reamostragem isotrópica, recorte de ROI
  features/               # 5 famílias de descritores + agregação corte->exame
  models/                  # baseline trivial + 6 modelos clássicos (Pipelines)
  evaluation/               # protocolo (StratifiedGroupKFold) + métricas
scripts/               # scripts executáveis (CLI) que usam src/
notebooks/             # EDA
tests/                  # smoke test de ponta a ponta (pytest)
references/            # fichamento de referências (references/fichamento.md)
article/                # artigo científico (main.tex, references.bib, main.pdf)
docs/                    # Anexo A — tabela de contribuição individual
data/raw/, data/sample/  # dados (não versionados — ver .gitignore)
outputs/                 # tabelas/figuras/modelos gerados (não versionados)
```

## 6. Reprodutibilidade

- Semente fixa em `src/config.py::SEED = 42`, propagada a todos os modelos e
  ao `StratifiedGroupKFold`.
- Nenhum caminho absoluto no código — tudo relativo à raiz do repositório via
  `src/config.py::ROOT_DIR`.
- Todo pré-processamento que aprende parâmetros dos dados (normalização, PCA,
  seleção de características, SMOTE) está dentro de `sklearn.Pipeline`,
  ajustado somente na partição de treino de cada dobra.
- Dados brutos não são versionados (`data/raw/`, `data/sample/` estão no
  `.gitignore`); apenas o código que os gera/consome está no repositório.
- Ambiente testado: Python 3.11, dependências em `requirements.txt`.

## 7. Uso de IA generativa

Partes deste repositório (estruturação inicial do pipeline, código de
extração de características e modelagem, redação de trechos do artigo e o
fichamento crítico de referências) foram produzidas com apoio de IA
generativa (Claude), a pedido do grupo, dada a natureza extensa do trabalho e
o prazo. Todo o código foi executado e validado (testes de fumaça, ver
`tests/`) e todo o texto foi revisado pelo grupo, que assume responsabilidade
integral por sua correção — **incluindo a verificação bibliográfica das
referências citadas em `references/fichamento.md` e `article/references.bib`**,
já que a IA não teve acesso à internet neste ambiente para checagem cruzada
via DOI/CrossRef. Esta declaração também consta em nota de rodapé no artigo
(`article/main.tex`), conforme exigido pelo enunciado (§6).

## 8. Pendências antes da entrega final (checklist do grupo)

- [ ] Cadastro e aceite dos termos de uso do desafio no Kaggle (todos os integrantes).
- [ ] Definir e documentar o critério de amostragem estratificada sobre os dados reais.
- [ ] Gerar o manifesto real e rodar `extract_features.py` + `run_experiment.py`.
- [ ] Substituir a Tabela 1 e inserir as Figuras (ROC/PR, matriz de confusão) reais no artigo.
- [ ] Adicionar análise de erro qualitativa e importância de características (permutação).
- [ ] Verificar bibliograficamente cada referência de `references/fichamento.md` (DOI/páginas).
- [ ] Expandir para o mínimo de 10 leituras fichadas conforme a pesquisa avançar (Semanas 2–3).
- [ ] Preencher e assinar o Anexo A (`docs/anexo_a_contribuicao.md`), gerar o PDF separado.
- [ ] Testar reprodutibilidade em máquina/ambiente limpo, seguindo só este README.
- [ ] Publicar o repositório (ou liberar acesso ao professor) e conferir o link no artigo.
- [ ] Conferir limite de 4 páginas do artigo após inserir os resultados reais.
