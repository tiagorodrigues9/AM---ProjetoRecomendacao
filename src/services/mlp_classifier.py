# src/services/mlp_classifier.py
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


class MusicMLPClassifier:
    """
    Serviço de classificação usando Rede Neural Multicamada (MLP).
    Prevê se uma música será 'Curtida' (1) ou 'Não Curtida' (0).
    """

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "mlp_model.joblib"
        self.model: MLPClassifier | None = None
    
        def train(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        hidden_layers: tuple = (64, 32),
        max_iter: int = 300,
        random_state: int = 42,
    ) -> "MusicMLPClassifier":
            print(f"[MLP] Treinando rede neural com arquitetura {hidden_layers}...")
            print(f"[MLP] Features: {X_train.shape[1]} | Amostras: {X_train.shape[0]}")

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=random_state,
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )
        self.model.fit(X_train, y_train)
        self.save()
        print(f"[MLP] Treinamento concluído em {self.model.n_iter_} épocas.")
        return self
    def save(self):
        """Salva o modelo treinado em disco usando joblib."""
        joblib.dump(self.model, self.model_path)
        print(f"[MLP] Modelo salvo em: {self.model_path}")

    def load(self):
        """Carrega o modelo salvo do disco."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {self.model_path}. "
                "Rode 'uv run python scripts/train_mlp.py' primeiro."
            )
        self.model = joblib.load(self.model_path)
        print(f"[MLP] Modelo carregado de: {self.model_path}")
    def predict(self, X: np.ndarray | pd.DataFrame) -> dict:
        """
        Faz previsão para novas amostras.

        Retorna dict com:
        - 'predictions': lista de 0/1
        - 'probabilities': probabilidade da classe positiva (liked=1)
        - 'labels': texto legível ('Curtida' ou 'Não Curtida')
        """
        if self.model is None:
            self.load()

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        prob_liked = probabilities[:, 1]

        labels = ["Curtida" if p == 1 else "Não Curtida" for p in predictions]

        return {
            "predictions": predictions.tolist(),
            "probabilities": prob_liked.tolist(),
            "labels": labels,
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Avalia o modelo no conjunto de teste.

        Retorna dict com:
        - 'accuracy': acurácia (float entre 0 e 1)
        - 'report': classification_report completo (string)
        """
        if self.model is None:
            self.load()

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=["Não Curtida", "Curtida"]
        )

        return {"accuracy": acc, "report": report}
    
    