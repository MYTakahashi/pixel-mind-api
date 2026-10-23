"""Carregamento dos modelos, pré-processamento e inferência."""
# Bibliotecas
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import mlflow
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import accuracy_score, f1_score
# Configuração dos caminhos raiz e criando diretórios para salvar modelos e métricas.
ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
MODEL_INFO_PATH = ROOT_DIR / "models" / "model_uris.json"

class ImageInputError(ValueError):
    """Imagem inválida ou impossível de processar."""

class ModelService:
    # Iniciando o serviço de modelos, carregando os modelos e métricas do MLflow.
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.model_uris = {}
    # Verifica se os modelos e métricas existem, caso contrário, solicita o treinamento.
    def load(self):
        if not MODEL_INFO_PATH.exists() or not METRICS_PATH.exists():
            raise RuntimeError(
                "Modelos não encontrados. Execute primeiro: python -m training.train"
            )
        # Rastreamento do MLflow usando banco de dados SQLite local.
        MLFLOW_DB = ROOT_DIR / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
        self.model_uris = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        # Carrega os modelos do MLflow, diferenciando entre modelos TensorFlow e Scikit-learn.
        for name, uri in self.model_uris.items():
            if name == "MLP":
                self.models[name] = mlflow.tensorflow.load_model(uri)
            else:
                self.models[name] = mlflow.sklearn.load_model(uri)
        return self

    @staticmethod
    def preprocess_image(raw: bytes):
        try:
            image = Image.open(io.BytesIO(raw))
            # Corrige a orientação da imagem com base nos metadados EXIF e converte para escala de cinza (L).
            image = ImageOps.exif_transpose(image).convert("L")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageInputError("O arquivo enviado não é uma imagem válida.") from exc
        # Valida as dimensões da imagem para evitar imagens muito pequenas ou muito grandes.
        if image.width < 10 or image.height < 10:
            raise ImageInputError("A imagem é muito pequena. Envie uma imagem com pelo menos 10 x 10 pixels.")
        if image.width * image.height > 25_000_000:
            raise ImageInputError("A imagem excede o limite de 25 megapixels.")

        arr = np.asarray(image, dtype=np.uint8)
        # MNIST: fundo preto e traço claro. Inverte fotos de papel/fundo claro.
        if float(arr.mean()) > 127:
            arr = 255 - arr

        # Remover ruído fraco e recortar o conteúdo desenhado.
        mask = arr > 35
        if not mask.any():
            return None, None, False

        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        crop = arr[y0:y1, x0:x1]
        h, w = crop.shape
        side = max(h, w)
        square = np.zeros((side, side), dtype=np.uint8)
        square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = crop
        resized = Image.fromarray(square).resize((20, 20), Image.Resampling.LANCZOS)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[4:24, 4:24] = np.asarray(resized, dtype=np.uint8)
        # Sinalizadores conservadores para rejeitar imagens vazias/sem conteúdo.
        occupancy = float((canvas > 35).mean())
        if occupancy < 0.015 or occupancy > 0.80:
            return canvas, canvas.astype("float32").reshape(1, 784) / 255.0, False
        return canvas, canvas.astype("float32").reshape(1, 784) / 255.0, True

    def predict(self, raw: bytes):
        canvas, vector, plausible = self.preprocess_image(raw)
        if canvas is None:
            return {
                "is_digit": False,
                "message": "Não foi possível identificar: imagem vazia ou irrreconhecível.",
                "processed_image_base64": None,
                "image_width": 28, "image_height": 28,
                "predictions": [],
                "note": "Envie uma imagem com um único dígito manuscrito, preferencialmente com fundo liso."
            }

        buffer = io.BytesIO()
        Image.fromarray(canvas).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        predictions = []
        # Itera sobre os modelos carregados, realizando previsões e calculando confiança e métricas de desempenho.
        for name in ("KNN", "Random Forest", "MLP"):
            model = self.models[name]
            if name == "MLP":
                probs = np.asarray(model.predict(vector, verbose=0))[0]
                digit = int(np.argmax(probs))
                confidence = float(probs[digit])
            else:
                digit = int(model.predict(vector)[0])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vector)[0]
                    confidence = float(np.max(probs))
                else:
                    confidence = 0.0
        # Armazena os resultados das previsões, incluindo métricas de desempenho para cada modelo.
            predictions.append({
                "model": name,
                "predicted_digit": digit,
                "confidence": confidence,
                "metrics": self.metrics[name],
            })

        # O classificador não foi treinado para detectar objetos fora do MNIST.
        # Confiança baixa entre os três modelos é um indício, não uma prova.
        mlp_conf = next(p["confidence"] for p in predictions if p["model"] == "MLP")
        agreement = len({p["predicted_digit"] for p in predictions}) == 1
        is_digit = bool(plausible and (mlp_conf >= 0.35 or agreement))
        message = "Dígito analisado com sucesso." if is_digit else "Não foi possível identificar com confiança se a imagem contém um número."
        return {
            "is_digit": is_digit,
            "message": message,
            "processed_image_base64": encoded,
            "image_width": 28, "image_height": 28,
            "predictions": predictions,
            "note": "O sistema usa regras simples para identificar o que não é número. "
                    "Ele pode confundir letras e objetos com números. "
                    "Para obter melhores resultados, envie apenas um único dígito escrito à mão."
        }
