"""Embute docs/model.json dentro de docs/index.html (página única, funciona no GitHub Pages ou abrindo o arquivo)."""
from pathlib import Path
R = Path(__file__).resolve().parents[1]
modelo = (R / "docs" / "model.json").read_text(encoding="utf-8").replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
html = (R / "tools" / "app_modelo.tpl").read_text(encoding="utf-8").replace("__MODEL_JSON__", modelo)
(R / "docs" / "index.html").write_text(html, encoding="utf-8")
print("docs/index.html", round(len(html) / 1024), "KB")
