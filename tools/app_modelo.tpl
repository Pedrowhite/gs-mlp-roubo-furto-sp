<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roubo ou Furto? · MLP · São Paulo</title>
<style>
  :root{--bg:#f6f4ef;--card:#fff;--ink:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--roubo:#c8452e;--furto:#2f6f8f;--accent:#1d1d1b}
  @media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1f1f1d;--ink:#f1efe8;--muted:#a3a19a;--line:#33322f;--accent:#f1efe8}}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  main{max-width:980px;margin:0 auto;padding:32px 20px 48px}
  header h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em} header p{margin:0;color:var(--muted)}
  .tag{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:99px;padding:2px 10px;margin:12px 6px 0 0;color:var(--muted)}
  .grid{display:grid;grid-template-columns:1.1fr .9fr;gap:20px;margin-top:24px} @media(max-width:760px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 14px;font-weight:600}
  label{display:block;font-size:13px;font-weight:600;margin:12px 0 4px}
  select{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .hint{font-size:12px;color:var(--muted);margin-top:4px}
  button{margin-top:18px;width:100%;padding:12px;border:0;border-radius:10px;background:var(--accent);color:var(--bg);font:600 15px system-ui;cursor:pointer}
  .verdict{font-size:28px;font-weight:700;margin:4px 0 14px;letter-spacing:-.01em}
  .bar{display:flex;height:40px;border-radius:8px;overflow:hidden;margin:6px 0 6px}
  .bar div{display:flex;align-items:center;padding:0 10px;color:#fff;font-weight:700;font-size:14px;white-space:nowrap;transition:width .4s}
  .legend{display:flex;justify-content:space-between;font-size:13px;color:var(--muted)}
  .note{font-size:13px;color:var(--muted);border-left:3px solid var(--line);padding-left:10px;margin-top:18px}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px} td{padding:5px 0;border-bottom:1px solid var(--line)} td:last-child{text-align:right;font-variant-numeric:tabular-nums}
  footer{margin-top:28px;font-size:12px;color:var(--muted)} a{color:inherit}
</style>
</head>
<body>
<main>
<header>
  <h1>Roubo ou furto?</h1>
  <p>Classificação de ocorrências de celular subtraído na cidade de São Paulo com uma rede neural MLP.</p>
  <span class="tag" id="tagModelo"></span><span class="tag">Dados: SSP-SP · Celulares Subtraídos 2025</span><span class="tag">Roda 100% no navegador</span>
</header>

<div class="grid">
  <section class="card">
    <h2>Características da ocorrência</h2>
    <div class="row">
      <div><label for="PERIODO">Período do dia</label><select id="PERIODO"></select><div class="hint" id="perHint"></div></div>
      <div><label for="DIA_SEMANA">Dia da semana</label><select id="DIA_SEMANA"></select></div>
    </div>
    <label for="TIPO_LOCAL">Tipo de local</label><select id="TIPO_LOCAL"></select>
    <label for="REGIAO">Região (seccional de polícia)</label><select id="REGIAO"></select>
    <label for="BAIRRO">Bairro</label><select id="BAIRRO"></select>
    <div class="hint">Só os 40 bairros mais frequentes na amostra; os demais foram agrupados em “Outros”.</div>
    <label for="MARCA">Marca do celular</label><select id="MARCA"></select>
    <button id="btn">Classificar</button>
  </section>

  <section class="card" aria-live="polite">
    <h2>Resultado da MLP</h2>
    <div class="verdict" id="verdict">—</div>
    <div class="bar"><div id="bR" style="background:var(--roubo);width:50%"></div><div id="bF" style="background:var(--furto);width:50%"></div></div>
    <div class="legend"><span>roubo</span><span>furto</span></div>
    <p class="note">Interpretação: <i>considerando que uma ocorrência aconteceu com essas características</i>, esta é a probabilidade que o modelo atribui a cada classe.
      <b>Não</b> é a chance de um crime acontecer nesse lugar ou horário.</p>
    <h2 style="margin-top:22px">Desempenho no conjunto de teste</h2>
    <table id="tabMet"></table>
  </section>
</div>
<footer>Global Solution · Turma 2CCPA · Pedro César de Camargo Branco (RM567971), Renan Rodrigues de Matos Almeida (RM567952), Thiago Garcia Batista Santos (RM568107)<br>Fonte: <a href="https://www.ssp.sp.gov.br/estatistica/consultas" target="_blank" rel="noopener">SSP-SP – Portal de Estatísticas</a></footer>
</main>

<script>
(function(){
try {
const M = __MODEL_JSON__;
const CAT = M.entrada.categoricas, CATS = M.entrada.categorias;
const DIAS = ["SEGUNDA","TERCA","QUARTA","QUINTA","SEXTA","SABADO","DOMINGO"];
const bonito = s => s==="NAO INFORMADO"?"Não informado":s==="OUTROS"?"Outros":
  s.toLowerCase().replace(/(^|[\s(\/.-])([a-zà-ú])/g,(m,a,b)=>a+b.toUpperCase())
   .replace(/\bTerca\b/,"Terça").replace(/\bSabado\b/,"Sábado").replace(/\bManha\b/,"Manhã").replace(/\bPublica\b/,"Pública");

function preenche(id, vals, padrao){
  const el = document.getElementById(id);
  vals.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=bonito(v);el.appendChild(o)});
  if (padrao && vals.includes(padrao)) el.value = padrao;
}
preenche("PERIODO", ["MADRUGADA","MANHA","TARDE","NOITE"].filter(p=>CATS.PERIODO.includes(p)), "NOITE");
const FAIXA={MADRUGADA:"0h–5h59",MANHA:"6h–11h59",TARDE:"12h–17h59",NOITE:"18h–23h59"};
const perEl=document.getElementById("PERIODO");
const atualizaPer=()=>document.getElementById("perHint").textContent=FAIXA[perEl.value]||""; perEl.onchange=atualizaPer; atualizaPer();
preenche("DIA_SEMANA", DIAS.filter(d=>CATS.DIA_SEMANA.includes(d)), "SEXTA");
const ordena = a => [...a].sort((x,y)=>(x==="OUTROS"||x==="NAO INFORMADO")-(y==="OUTROS"||y==="NAO INFORMADO")||x.localeCompare(y));
["TIPO_LOCAL","REGIAO","BAIRRO","MARCA"].forEach(c=>preenche(c, ordena(CATS[c])));
const tl = CATS.TIPO_LOCAL.find(v=>v.includes("VIA PUBLICA")); if(tl) document.getElementById("TIPO_LOCAL").value=tl;

// Mesmo pré-processamento do notebook: one-hot de PERIODO, DIA_SEMANA, TIPO_LOCAL, REGIAO, BAIRRO, MARCA (mesma ordem do OneHotEncoder)
function vetor(){
  const x = [], val = {}; CAT.forEach(c=>{ val[c]=document.getElementById(c).value });
  CAT.forEach(c=>CATS[c].forEach(v=>x.push(v===val[c]?1:0)));
  return x;
}
const act = {relu:z=>Math.max(0,z), tanh:Math.tanh, sigmoid:z=>1/(1+Math.exp(-z)), linear:z=>z};
function forward(x){ // propagação direta camada a camada: a = f(xW + b)
  let a = x;
  for (const L of M.camadas){
    const out = new Array(L.b.length).fill(0);
    for (let j=0;j<L.b.length;j++){ let s=L.b[j]; for(let i=0;i<a.length;i++) s+=a[i]*L.W[i][j]; out[j]=act[L.ativacao](s); }
    a = out;
  }
  return a[0]; // P(roubo)
}
function classifica(){
  const p = forward(vetor()), r = Math.round(p*100), f = 100-r;
  document.getElementById("bR").style.width=r+"%"; document.getElementById("bR").textContent=r>=8?r+"%":"";
  document.getElementById("bF").style.width=f+"%"; document.getElementById("bF").textContent=f>=8?f+"%":"";
  const v=document.getElementById("verdict"); v.textContent = p>=0.5?"Mais compatível com ROUBO":"Mais compatível com FURTO";
  v.style.color = p>=0.5?"var(--roubo)":"var(--furto)";
}
document.getElementById("btn").onclick=classifica;
document.getElementById("tagModelo").textContent="Modelo: "+M.modelo;
const mt=M.metricas_teste, pct=v=>(v*100).toFixed(1)+"%";
document.getElementById("tabMet").innerHTML=[["Acurácia",mt.acuracia],["Precision (roubo)",mt.precision],["Recall (roubo)",mt.recall],["F1-score (roubo)",mt.f1]]
  .map(([k,v])=>`<tr><td>${k}</td><td>${pct(v)}</td></tr>`).join("");
classifica();
window.__mlp = {forward, vetor, M}; // usado no teste automático
} catch (e) {
  const v = document.getElementById("verdict");
  if (v) { v.textContent = "Erro ao carregar o modelo: " + e.message; v.style.fontSize = "16px"; }
}
})();
</script>
</body>
</html>
