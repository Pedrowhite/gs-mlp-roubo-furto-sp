# ==============================================================
# Extração dos dados da SSP-SP (rodar no Google Colab)
# Baixa a base "Celulares Subtraídos 2025", mantém só as colunas
# úteis e só os registros com relação à capital, e gera um arquivo
# pequeno (ssp_capital_2025.csv.gz) para o restante do trabalho.
# ==============================================================
import os, unicodedata, pandas as pd

URL = ("https://www.ssp.sp.gov.br/assets/estatistica/transparencia/"
       "baseDados/celularesSub/CelularesSubtraidos_2025.xlsx")
ARQ = "CelularesSubtraidos_2025.xlsx"

if not os.path.exists(ARQ):
    print("Baixando da SSP-SP (~100 MB)...")
    ok = os.system(f'wget -q --user-agent="Mozilla/5.0" -O {ARQ} "{URL}"') == 0
    if not ok or os.path.getsize(ARQ) < 1_000_000:
        print("Download automático falhou. Baixe o arquivo pelo navegador no link abaixo")
        print("e envie aqui quando aparecer o botão 'Escolher arquivos':\n", URL)
        from google.colab import files
        up = files.upload()
        ARQ = list(up.keys())[0]

print("Lendo a planilha (pode levar 3-5 minutos)...")
xls = pd.ExcelFile(ARQ, engine="openpyxl")
print("Abas:", xls.sheet_names)
aba = max(xls.sheet_names, key=lambda s: ("CELULAR" in s.upper(), len(s)))
df = pd.read_excel(xls, sheet_name=aba, dtype=str)
print("Formato original:", df.shape)

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.strip().upper()
df.columns = [norm(c) for c in df.columns]

QUERO = ["NUM_BO", "ANO_BO", "ID_DELEGACIA", "NOME_DELEGACIA", "CIDADE", "NOME_MUNICIPIO",
         "NOME_MUNICIPIO_CIRC", "NOME_SECCIONAL_CIRC", "NOME_DEPARTAMENTO_CIRC",
         "NOME_DELEGACIA_CIRC", "RUBRICA", "DESCR_CONDUTA", "NATUREZA_APURADA",
         "DATA_OCORRENCIA_BO", "HORA_OCORRENCIA", "HORA_OCORRENCIA_BO", "DESCR_PERIODO",
         "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL", "BAIRRO", "LATITUDE", "LONGITUDE",
         "DESCR_TIPO_OBJETO", "DESCR_MODO_OBJETO", "MARCA_OBJETO", "QUANTIDADE_OBJETO",
         "FLAG_FLAGRANTE", "AUTORIA_BO", "DESDOBRAMENTO"]
cols = [c for c in QUERO if c in df.columns]
print("Colunas mantidas:", cols)
print("Todas as colunas do arquivo:", list(df.columns))

cap = pd.Series(False, index=df.index)
for c in ["CIDADE", "NOME_MUNICIPIO_CIRC", "NOME_MUNICIPIO"]:
    if c in df.columns:
        cap |= df[c].map(norm).isin(["S.PAULO", "SAO PAULO", "S. PAULO"])
out = df.loc[cap, cols]
print("Registros ligados à capital:", len(out))
out.to_csv("ssp_capital_2025.csv.gz", index=False, compression="gzip")
print("Arquivo gerado:", os.path.getsize("ssp_capital_2025.csv.gz") / 1e6, "MB")

from google.colab import files
files.download("ssp_capital_2025.csv.gz")
