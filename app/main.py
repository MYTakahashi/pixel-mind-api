# Biblotecas
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.routers import router
from app.services import ModelService

import webbrowser

# Instancia o serviço de modelos
model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega os três modelos uma única vez no início da aplicação.
    model_service.load()
    app.state.model_service = model_service

    webbrowser.open("http://127.0.0.1:8000", new=2)  # Abre a página inicial no navegador padrão

    yield
    # Liberação de recursos (não é necessário descarregar explicitamente os modelos).
    model_service.models.clear()
# Inicia a aplicação FastAPI com informações de título, descrição e versão, e inclui o roteador.
app = FastAPI(
    title="Pixel Mind API",
    description="API para classificar dígitos manuscritos usando KNN, Random Forest e MLP treinados com MNIST.",
    version="1.0.0",
    lifespan=lifespan,
)
# Registra as rotas principais
app.include_router(router)

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def home():
    """Interface simples para enviar uma imagem e visualizar o resultado."""
    return HTML_PAGE

HTML_PAGE = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pixel Mind — Classificador de dígitos</title>
<style>
body{font-family:Arial,sans-serif;max-width:900px;margin:32px auto;padding:0 18px;color:#20252b}
h1{color:#2d6a4f}button{padding:10px 16px;background:#2d6a4f;color:white;border:0;border-radius:6px;cursor:pointer}
input{margin:14px 0}#status{margin:16px 0;font-weight:bold}.layout{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
img{image-rendering:pixelated;border:1px solid #ddd;width:196px;height:196px;object-fit:contain}
table{border-collapse:collapse;width:100%;margin-top:18px}th,td{padding:10px;border:1px solid #ddd;text-align:left}
th{background:#f2f5f3}.muted{color:#5c6670;font-size:.92rem}
</style></head>
<body>
<h1>Pixel Mind API</h1>
<p>Envie uma imagem com um único dígito manuscrito. Os três modelos serão comparados.</p>
<form id="form"><input id="file" name="file" type="file" accept="image/png,image/jpeg,image/bmp,image/webp" required>
<br><button type="submit">Clique para analisar a imagem</button></form>
<div id="status" role="status"></div>
<div id="results"></div>
<p class="muted">As métricas de desempenho (Acurácia e F1-Score) referem-se a base padrão de testes MNIST.</p>
<script>
const form=document.querySelector('#form'),status=document.querySelector('#status'),results=document.querySelector('#results');
form.addEventListener('submit',async e=>{
 e.preventDefault(); const file=document.querySelector('#file').files[0]; if(!file)return;
 status.textContent='Analisando imagem...'; results.innerHTML='';
 const data=new FormData(); data.append('file',file);
 try{
  const response=await fetch('/predict',{method:'POST',body:data});
  const body=await response.json();
  if(!response.ok){status.textContent='Erro ('+response.status+'): '+(body.detail||'Não foi possível processar a imagem.');return;}
  status.textContent=body.message;
  let html='<div class="layout">';
  if(body.processed_image_base64) html+='<div><h3>Imagem processada (28 × 28)</h3><img alt="Imagem processada" src="data:image/png;base64,'+body.processed_image_base64+'"></div>';
  if(body.predictions.length){
   html+='<div style="flex:1;min-width:280px"><h3>Previsões dos modelos</h3><table><thead><tr><th>Modelo</th><th>Número</th><th>Confiança</th><th>Acurácia teste</th><th>F1-Score</th></tr></thead><tbody>';
   for(const p of body.predictions) html+='<tr><td>'+p.model+'</td><td><strong>'+p.predicted_digit+'</strong></td><td>'+(p.confidence*100).toFixed(1)+'%</td><td>'+(p.metrics.accuracy*100).toFixed(2)+'%</td><td>'+(p.metrics.f1_score*100).toFixed(2)+'%</td></tr>';
   html+='</tbody></table></div>';
  }
  html+='</div>'; if(body.note)html+='<p class="muted">'+body.note+'</p>'; results.innerHTML=html;
 }catch(err){status.textContent='Não foi possível conectar à API. Verifique se o servidor está em execução.';}
});
</script></body></html>"""