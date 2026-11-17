# Roubo ou furto? Classificação de ocorrências em São Paulo com MLP

Global Solution · Aprendizado Profundo com Redes Neurais Convolucionais

Uma rede neural **MLP** recebe as características de uma ocorrência de **celular subtraído na cidade de São Paulo**
(período do dia, dia da semana, tipo de local, região, bairro e marca do aparelho) e estima se o registro é mais
compatível com **roubo** (art. 157) ou **furto** (art. 155).

> O modelo **não** prevê se um crime vai acontecer. Ele responde: *dado que uma ocorrência aconteceu com essas
> características, qual classe é mais compatível?*

## 🔗 Aplicação web

**https://github.com/Pedrowhite/gs-mlp-roubo-furto-sp**

A página roda a MLP treinada direto no navegador (sem servidor). Os pesos exportados do Keras ficam embutidos no HTML.

## Integrantes · Turma 2CCPA

| Nome | RM |
|---|---|
| Pedro César de Camargo Branco | RM567971 |
| Renan Rodrigues de Matos Almeida | RM567952 |
| Thiago Garcia Batista Santos | RM568107 |

## Dados

| Item | Detalhe |
|---|---|
| Fonte pública | Secretaria da Segurança Pública de SP (SSP-SP), Portal de Estatísticas → Consultas → Bases de Dados → **Celulares Subtraídos** — https://www.ssp.sp.gov.br/estatistica/consultas |
| Arquivo original | `CelularesSubtraidos_2025.xlsx` — https://www.ssp.sp.gov.br/assets/estatistica/transparencia/baseDados/celularesSub/CelularesSubtraidos_2025.xlsx |
| Período | Boletins de 2025 |
| Recorte | `CIDADE == "S.PAULO"` (cidade **do fato**, não do registro do BO; 99,9% de concordância com o município da circunscrição) |
| Rótulo | `RUBRICA`: Roubo (art. 157) → 1 · Furto (art. 155) → 0 · demais naturezas excluídas |
| Duplicidades | 1 linha por boletim (`NUM_BO` + `ANO_BO` + `ID_DELEGACIA`); BOs com roubo **e** furto excluídos |
| Volume | 216.129 itens na capital → 161.071 boletins válidos → **amostra de 20.000** (semente 42; 37% roubo) |

### Como obter os dados
1. Abra o Google Colab e rode `tools/colab_extrair_dados.py`. Ele baixa a planilha da SSP (~100 MB), mantém as colunas
   de interesse e as linhas ligadas à capital, e gera `ssp_capital_2025.csv.gz`.
2. Coloque esse arquivo em `data/` (o arquivo já está incluído neste repositório).

## Atributos de entrada

| Atributo | Coluna de origem |
|---|---|
| Período do dia | `HORA_OCORRENCIA` (convertida em período) ou `DESCR_PERIODO` |
| Dia da semana | `DATA_OCORRENCIA_BO` |
| Tipo de local | `DESCR_SUBTIPOLOCAL` (12 mais frequentes + Outros) |
| Região | `NOME_SECCIONAL_CIRC` |
| Bairro | `BAIRRO` (40 mais frequentes + Outros) |
| Marca | `MARCA_OBJETO` |

Ficaram de fora a natureza, a conduta, o modo do objeto, o desdobramento, o flagrante e a autoria (todos revelam ou
derivam a classe, ou seja, vazamento), além do endereço exato, das coordenadas e de dados pessoais.
**A hora exata também ficou fora:** ela só existe em metade dos BOs, e isso depende do tipo de registro
(67% de roubo com hora contra 8% sem). A rede aprenderia o formato do boletim, e não a ocorrência.

## Resultados

| Modelo | Camadas ocultas | F1 val. | Acurácia val. |
|---|---|---|---|
| **M1 (escolhido)** | 16 ReLU | **0,637** | 70,5% |
| M2 | 64 → 32 ReLU + Dropout | 0,633 | 69,7% |
| M3 | 128 → 64 → 32 ReLU + Dropout + L2 | 0,637 | 70,6% |
| M4 | 256 → 128 → 64 sem regularização | 0,563 | 66,0% |
| M5 | 64 → 32 tanh + Dropout | 0,628 | 68,7% |

**Teste (M1, 3.000 BOs):** acurácia 67,4% · precision 55,0% · recall 65,4% · F1 0,597
(referência "sempre furto": acurácia 63,0%, F1 0).

## Estrutura

```
├── notebooks/MLP_Roubo_Furto_SP.ipynb   # notebook completo (já executado)
├── data/ssp_capital_2025.csv.gz         # dados da capital extraídos da SSP-SP
├── modelo/                              # modelo Keras treinado, tabela comparativa, resumo
├── docs/index.html                      # aplicação web (GitHub Pages)
├── docs/model.json                      # pesos e categorias exportados
├── figs/                                # gráficos usados no relatório
├── relatorio/                           # relatório técnico (10 slides)
└── tools/                               # extração no Colab, gerador do app, teste do app
```

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/MLP_Roubo_Furto_SP.ipynb   # Run All: treina, avalia e exporta docs/model.json
python tools/build_app.py                             # embute o modelo em docs/index.html
```

O app também funciona abrindo **`docs/index.html`** direto no navegador (é o único arquivo do app; `tools/app_modelo.tpl` é só o molde usado para gerá-lo).
O notebook também roda no Google Colab: suba a pasta do projeto e execute todas as células.

## Uso de IA generativa

Usamos o Claude (Anthropic) para estruturar o código, depurar, construir a aplicação web e redigir a documentação.
A equipe revisou as decisões, executou o notebook de ponta a ponta e conferiu automaticamente que o app reproduz a
saída do Keras (`tools/verifica_app.py`, diferença máxima de 6×10⁻⁷).
