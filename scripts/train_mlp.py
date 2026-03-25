# scripts/train_mlp.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.mlp_classifier import MusicMLPClassifier


def main():
    features_path = project_root / "data" / "processed" / "dataset_features.csv"
    clean_path    = project_root / "data" / "processed" / "dataset_clean.csv"

    print("=" * 60)
    print("  [MLP] Treinamento da Rede Neural")
    print("=" * 60)

    if not features_path.exists():
        print(f"\n[ERRO] Arquivo não encontrado: {features_path}")
        print("DICA: Rode 'uv run python scripts/train_feature_engineer.py' primeiro.")
        return

    if not clean_path.exists():
        print(f"\n[ERRO] Arquivo não encontrado: {clean_path}")
        return

    print("\n[OK] Pré-requisitos verificados.")

        # --- 1. Carregar features ---
    print("\n[1/5] Carregando dataset de features...")
    X = pd.read_csv(features_path)
    print(f"       Shape: {X.shape}")
    print(f"       Primeiras colunas: {list(X.columns[:5])}...")

        # --- 2. Criar o label (Y) ---
    print("\n[2/5] Criando label 'liked'...")
    df_clean = pd.read_csv(clean_path)

    mediana = df_clean["popularity"].median()
    y = (df_clean["popularity"] > mediana).astype(int)

    print(f"       Mediana de popularity: {mediana}")
    print(f"       Curtidas (1): {y.sum()} | Não Curtidas (0): {len(y) - y.sum()}")

    # Garantir alinhamento entre X e y
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]

        # --- 3. Divisão Treino / Teste ---
    print("\n[3/5] Dividindo em Treino (80%) e Teste (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"       Treino: {X_train.shape[0]} amostras")
    print(f"       Teste:  {X_test.shape[0]} amostras")

        # --- 4. Treinar ---
    print("\n[4/5] Treinando a Rede Neural MLP...")
    mlp = MusicMLPClassifier()
    mlp.train(X_train, y_train, hidden_layers=(64, 32), max_iter=300)

    # --- 5. Avaliar ---
    print("\n[5/5] Avaliando no conjunto de teste...")
    results = mlp.evaluate(X_test, y_test)

    print(f"\n{'=' * 60}")
    print(f"  RESULTADO FINAL")
    print(f"{'=' * 60}")
    print(f"  Acurácia: {results['accuracy']:.2%}")
    print(f"\n{results['report']}")
    print(f"\n[OK] Modelo salvo em: models/mlp_model.joblib")


if __name__ == "__main__":
    main()