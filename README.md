# TP1 — Baseline de Projeto de Pesquisa (Grupo G2)

**Disciplina:** Tópicos Especiais em Sistemas de Informação
**Desafio RSNA atribuído:** 2020 — *Pulmonary Embolism Detection*
**Integrantes:** Rawenne da Silva Leite · Wheverson de Abreu Lima · João Victor do Nascimento Silva

Este repositório contém o baseline clássico (sem redes neurais profundas) construído
para o desafio RSNA STR Pulmonary Embolism Detection: leitura/pré-processamento de
DICOM, extração de cinco famílias de características hand-crafted, sete modelos
clássicos (incluindo o baseline trivial obrigatório) e um protocolo experimental
sem vazamento de dados (partição por paciente).

## ⚠️ Status atual: piloto real rodado (N=46), ainda pequeno para ser um benchmark

O ambiente onde este repositório foi originalmente montado **não tinha acesso ao
Kaggle** (rede corporativa bloqueava `kaggle.com`). Para contornar isso, a amostra
real foi extraída **em um Kaggle Notebook** (o dado já reside no servidor do Kaggle,
sem precisar baixar o conjunto completo — ~750 GB — para uma máquina local) e
transferida em lotes de zip pequenos (dentro do limite de upload do canal usado).

**O que já foi feito com dados reais (não sintéticos):**
- 46 exames reais do RSNA STR PE Dataset (23 positivos + 23 negativos), extraídos
  em três rodadas (6 exames/15 cortes, semente 42; +16 exames/6 cortes, semente 43;
  +24 exames/4 cortes, semente 44), amostrados de forma aleatória estratificada por
  classe. Critério documentado em `data/raw/real_sample_manifest.json` (não
  versionado — ver `.gitignore` — mas reproduzível a partir do `train.csv` oficial).
- Extração real das 5 famílias de características e execução real dos 7 modelos,
  com ajuste de hiperparâmetros em validação aninhada e 5 dobras
  (`scripts/extract_features.py` + `scripts/run_experiment.py --n-splits 5 --tune`).
- Os números e a figura no artigo (`article/main.tex`, Seção 4) **são reais**,
  não fabricados — vêm de `outputs/tables/results_summary.csv` desta execução.
- Alguns arquivos DICOM usam compressão JPEG Lossless, exigindo `pylibjpeg` +
  `pylibjpeg-libjpeg` (já em `requirements.txt`).

**O que isso já mostra, com mais confiança que os pilotos anteriores:** um piso de
desempenho modesto e honesto — o melhor modelo (Random Forest) atinge apenas
AUC-ROC≈0,61, e metade dos modelos fica abaixo do baseline trivial por
superajuste. Isso é consistente com o que se espera de características
hand-crafted sem segmentação vascular sobre poucas dezenas de exames — exatamente
o tipo de "piso a ser superado" que o TP pede. **Ainda não é robusto o bastante
para conclusões definitivas**; ampliar a amostra continua sendo a prioridade
seguinte (ver checklist ao final).

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

## 2. Como foi obtida a amostra real (e como ampliá-la)

Como o dataset completo (~750 GB) é inviável de baixar em máquina pessoal, e o
ambiente de desenvolvimento também não tinha acesso ao Kaggle, o caminho usado foi:

1. Aceitar os termos de uso da competição em
   https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/rules
   (cadastro/aceite é responsabilidade de cada integrante, conforme §2 do enunciado).
2. Criar um **Kaggle Notebook** a partir da página da competição (aba **Code → New
   Notebook**) — o dataset fica disponível em
   `/kaggle/input/competitions/rsna-str-pulmonary-embolism-detection/` sem download
   local algum.
3. Rodar, dentro do notebook, um script que: lê `train.csv`, agrupa por
   `StudyInstanceUID` (rótulo = `1 - negative_exam_for_pe`), seleciona uma amostra
   estratificada por classe com semente fixa, copia um subconjunto de cortes por
   exame para `/kaggle/working/`, e compacta em zip.
4. Baixar apenas esse zip pequeno (não o dataset inteiro) e trazer para este
   repositório em `data/raw/`.

**Para ampliar a amostra atual (N=46) ainda mais**, repita o mesmo processo
aumentando `N_POSITIVE`/`N_NEGATIVE` no notebook (excluindo os `StudyInstanceUID`
já usados, listados em `data/raw/real_sample_manifest.json`, para não duplicar) — o
script já usado neste histórico gera o zip, verifica o tamanho antes de baixar e
ajusta automaticamente o número de cortes por exame para caber no limite de
transferência disponível. Quanto maior a amostra, menos superajuste e mais
representativo o protocolo de validação cruzada (já usando 5 dobras e ajuste de
hiperparâmetros, mas ainda com poucos exemplos por dobra).

O manifesto de um lote de amostra real deve seguir o formato de
`data/sample/synthetic_manifest.json` (lista de objetos com `patient_id`,
`study_instance_uid`, `exam_dir`, `pe_present_on_exam`).

## 3. Rodando o pipeline completo

```bash
# 0) (opcional) gerar dados sintéticos de exemplo, só para validar o pipeline
python scripts/generate_synthetic_sample.py

# 1) extrair características (amostra real de 46 exames já usada no artigo)
python scripts/extract_features.py \
    --manifest data/raw/real_sample_manifest.json \
    --data-root data/raw/real_sample_pilot \
    --out outputs/tables/features_real_pilot.parquet

# 2) treinar e avaliar todos os modelos (baseline trivial + 7 clássicos), com tuning
python scripts/run_experiment.py --features outputs/tables/features_real_pilot.parquet --n-splits 5 --tune
```

Saídas geradas:
- `outputs/tables/results_summary.csv` — tabela comparativa descritor×modelo
  (média ± desvio entre dobras) — é a fonte da Tabela 1 do artigo.
- `outputs/figures/roc_<modelo>.png`, `pr_<modelo>.png`,
  `confusion_matrix_<modelo>.png` — para o melhor modelo (maior AUC-ROC médio).
- `outputs/models/<modelo>.joblib` — modelos finais serializados.

Ao rodar sobre uma amostra maior, copie os números atualizados de
`outputs/tables/results_summary.csv` para a Tabela 1 de `article/main.tex`, troque
a figura em `article/figures/roc_<modelo>.png` pela nova, e recompile
(`pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`,
conferindo o limite de 4 páginas).

## 4. EDA

`notebooks/01_eda.ipynb` — distribuição de classes, metadados DICOM, exemplos
visuais por classe. Por padrão lê o manifesto sintético; troque
`MANIFEST_PATH`/`DATA_ROOT` no notebook para `data/raw/real_sample_manifest.json`
para usar a amostra real.

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
article/                # artigo científico (main.tex, references.bib, main.pdf, figures/)
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
- Dados brutos (sintéticos ou reais) não são versionados (`data/raw/`,
  `data/sample/` estão no `.gitignore`, inclusive por respeito aos termos de uso
  do Kaggle, que proíbem redistribuir os dados da competição); apenas o código que
  os gera/consome, e os resultados agregados (tabelas/figuras publicadas no
  artigo), estão no repositório.
- Ambiente testado: Python 3.11, dependências em `requirements.txt`.

## 7. Checklist do grupo antes da entrega final

- [x] Cadastro e aceite dos termos de uso do desafio no Kaggle.
- [x] Amostra real extraída e documentada (piloto, N=46, 23+23).
- [x] Pipeline executado de ponta a ponta sobre dados reais; resultados reais no artigo.
- [x] Validação cruzada com 5 dobras e ajuste de hiperparâmetros (`--tune`).
- [x] Ao menos uma figura (curva ROC) e análise de erro qualitativa no artigo.
- [x] Limite de 4 páginas do artigo (conferido após inserir os resultados reais).
- [ ] **Ampliar a amostra real ainda mais** (prioridade nº 1 — 46 exames para 8
      modelos e ~500 características ainda causa superajuste visível em alguns
      modelos; repetir o processo da Seção 2 com `N_POSITIVE`/`N_NEGATIVE`
      maiores, excluindo os `StudyInstanceUID` já usados).
- [ ] Reexecutar `extract_features.py` + `run_experiment.py --n-splits 5 --tune`
      sobre a amostra ampliada e atualizar Tabela 1/Figura 1 do artigo.
- [x] Verificar bibliograficamente cada referência (DOI/páginas/título) — 1 correção feita (título do PENet).
- [ ] Expandir para leituras adicionais conforme a pesquisa avançar (Semanas 2–3).
- [ ] Preencher e assinar o Anexo A (`docs/anexo_a_contribuicao.md`), gerar o PDF separado.
- [ ] Testar reprodutibilidade em máquina/ambiente limpo, seguindo só este README.
- [ ] Confirmar que o link do repositório no artigo está correto e acessível ao professor.
