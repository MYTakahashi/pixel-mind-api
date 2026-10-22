"""Treina os três modelos, calcula métricas e registra no MLflow."""
#Bibliotecas necessárias: scikit-learn, tensorflow, mlflow, skops, uvicorn
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Configuração dos caminhos raiz e criando diretórios para salvar modelos e métricas.
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
TRACKING_DIR = ROOT / "mlruns"
TRACKING_DIR.mkdir(exist_ok=True)

def main():
    # Para garantir reprodutibilidade, definindo sementes para NumPy e TensorFlow.
    np.random.seed(42)
    tf.random.set_seed(42)
    # Mesmo conjunto usado no notebook de origem: MNIST 70.000 x 784.
    print("Baixando/carregando MNIST do OpenML, na primeira execução pode demorar um pouco...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype("float32")
    y = mnist.target.astype("int64")
    # 70% treino, 10% validação, 20% teste.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    # Normalizando os dados para o intervalo [0, 1] usando MinMaxScaler.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")
    print(f"Treino: {len(y_train):,} | validação: {len(y_val):,} | teste: {len(y_test):,}")
    # Criando o banco de dados SQLite para o MLflow, se não existir, e configurando o rastreamento do MLflow.
    MLFLOW_DB = ROOT / "mlflow.db"

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("pixel-mind-mnist")
    metrics = {}
    model_uris = {}
    # Iniciando uma execução do MLflow para registrar os modelos e métricas.
    with mlflow.start_run(run_name="pixel-mind-models") as run:
        # KNN
        print("Treinando modelo KNN...")
        knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        start = time.perf_counter()
        knn.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        pred = knn.predict(X_test)
        metrics["KNN"] = {"accuracy": float(accuracy_score(y_test, pred)),
                          "f1_score": float(f1_score(y_test, pred, average="weighted")),
                          "dataset": "MNIST - conjunto de teste"}
        mlflow.log_params({"knn_n_neighbors": 5, "knn_weights": "distance"})
        mlflow.log_metrics({"knn_accuracy": metrics["KNN"]["accuracy"],
                            "knn_f1_weighted": metrics["KNN"]["f1_score"],
                            "knn_train_seconds": train_seconds})
        mlflow.sklearn.log_model(knn, artifact_path="knn")
        model_uris["KNN"] = f"runs:/{run.info.run_id}/knn"
        # Random Forest
        print("Treinando modelo Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        start = time.perf_counter()
        rf.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        pred = rf.predict(X_test)
        metrics["Random Forest"] = {"accuracy": float(accuracy_score(y_test, pred)),
                                    "f1_score": float(f1_score(y_test, pred, average="weighted")),
                                    "dataset": "MNIST - conjunto de teste"}
        mlflow.log_params({"rf_n_estimators": 100, "rf_max_depth": 20})
        mlflow.log_metrics({"rf_accuracy": metrics["Random Forest"]["accuracy"],
                            "rf_f1_weighted": metrics["Random Forest"]["f1_score"],
                            "rf_train_seconds": train_seconds})
        mlflow.sklearn.log_model(
            rf,
            artifact_path="random_forest",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        model_uris["Random Forest"] = f"runs:/{run.info.run_id}/random_forest"
        # rede neural MLP - Multi-Layer Perceptron
        print("Treinando modelo de rede neural MLP (15 épocas)...")
        mlp = keras.Sequential([
            layers.Input(shape=(784,)),
            layers.Dense(128, activation="relu", name="Camada_Oculta_1"),
            layers.Dense(64, activation="relu", name="Camada_Oculta_2"),
            layers.Dense(10, activation="softmax", name="Camada_Saida"),
        ])
        mlp.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        start = time.perf_counter()
        mlp.fit(X_train, y_train, epochs=15, batch_size=64,
                validation_data=(X_val, y_val), verbose=1)
        train_seconds = time.perf_counter() - start
        # Realiza a predição no conjunto de teste e calcula métricas de avaliação com o argmax.
        pred = np.argmax(mlp.predict(X_test, batch_size=256, verbose=0), axis=1)
        metrics["MLP"] = {"accuracy": float(accuracy_score(y_test, pred)),
                          "f1_score": float(f1_score(y_test, pred, average="weighted")),
                          "dataset": "MNIST - conjunto de teste"}
        mlflow.log_params({"mlp_hidden_units": "128,64", "mlp_activation": "relu",
                           "mlp_learning_rate": 0.001, "mlp_epochs": 15, "mlp_batch_size": 64})
        mlflow.log_metrics({"mlp_accuracy": metrics["MLP"]["accuracy"],
                            "mlp_f1_weighted": metrics["MLP"]["f1_score"],
                            "mlp_train_seconds": train_seconds})
        mlflow.tensorflow.log_model(mlp, artifact_path="mlp")
        model_uris["MLP"] = f"runs:/{run.info.run_id}/mlp"
        # Registrando métricas e URIs dos modelos no MLflow.
        mlflow.log_dict(metrics, "evaluation/metrics.json")
        mlflow.log_dict(model_uris, "evaluation/model_uris.json")
    # Salvando métricas e URIs dos modelos em arquivos JSON no diretório MODELS_DIR.
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODELS_DIR / "model_uris.json").write_text(json.dumps(model_uris, indent=2), encoding="utf-8")
    print("\nTreinamento concluído. Métricas no teste:")
    for name, result in metrics.items():
        print(f"{name}: acurácia={result['accuracy']:.4f} | F1-Score={result['f1_score']:.4f}")
    print(f"\nModelos registrados no MLflow. URI do experimento: {mlflow.get_tracking_uri()}")
    print("Agora execute: uvicorn app.main:app --reload")

if __name__ == "__main__":
    main()
