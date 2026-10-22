# Bibliotecas
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app.schemas import HealthResponse, PredictionResponse
from app.services import ImageInputError, ModelService
# Instancia o roteador FastAPI para definir as rotas da API.
router = APIRouter()
# Tamanho máximo de upload.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

def get_service() -> ModelService:
    from app.main import model_service
    return model_service

@router.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
async def health():
    service = get_service()
    loaded = {name: name in service.models for name in ("KNN", "Random Forest", "MLP")}
    healthy = all(loaded.values())
    return {
        "status": "ok" if healthy else "unhealthy",
        "message": "API e modelos carregados." if healthy else "Um ou mais modelos não foram carregados.",
        "models_loaded": loaded,
    }

@router.post("/predict", response_model=PredictionResponse, tags=["Inferência"])
async def predict(file: UploadFile = File(..., description="Imagem PNG, JPEG ou BMP com um único dígito manuscrito.")):
    # Verifica se o arquivo é válido.
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome de arquivo não informado.")
    content_type = (file.content_type or "").lower()
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}
    if content_type not in allowed:
        raise HTTPException(status_code=415, detail="Formato não suportado. Envie PNG, JPEG, BMP ou WEBP.")
    # Lê o conteúdo do arquivo, limitando o tamanho máximo.
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Arquivo muito grande. O limite é 10 MB.")
    # Executa o serviço de predição e retorna o resultado.
    try:
        return get_service().predict(raw)
    # Trata erros relacionados à entrada de imagem, retornando um erro HTTP 400.
    except ImageInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Trata erros de execução, como modelos não carregados, retornando um erro HTTP 503.
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Evita expor detalhes internos ao cliente.
        raise HTTPException(status_code=500, detail="Erro interno durante a inferência.") from exc