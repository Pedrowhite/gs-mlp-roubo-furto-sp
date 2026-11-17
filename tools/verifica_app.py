"""Confere que o app web (JavaScript) dá a mesma probabilidade que o Keras e tira um print da interface."""
import json, math
from pathlib import Path
from playwright.sync_api import sync_playwright
R = Path(__file__).resolve().parents[1]
ex = json.loads((R / "modelo" / "exemplos_verificacao.json").read_text())
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome")
    pg = b.new_page(viewport={"width": 1100, "height": 820}, device_scale_factor=2)
    pg.goto((R / "docs" / "index.html").as_uri())
    erro = 0
    for e in ex:
        for c in ["PERIODO", "DIA_SEMANA", "TIPO_LOCAL", "REGIAO", "BAIRRO", "MARCA"]:
            pg.select_option(f"#{c}", e[c])
        pj = pg.evaluate("() => window.__mlp.forward(window.__mlp.vetor())")
        erro = max(erro, abs(pj - e["p_roubo_keras"]))
        print(f"keras={e['p_roubo_keras']:.5f}  js={pj:.5f}")
    print("maior diferença:", erro); assert erro < 1e-3
    pg.reload(); pg.wait_for_timeout(300)
    (R / "figs").mkdir(exist_ok=True); pg.screenshot(path=str(R / "figs" / "app.png"), full_page=True)
    b.close()
