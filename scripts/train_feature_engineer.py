import sys
import pandas as pd
from pathlib import Path

# Adiciona a raiz do projeto no path do script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.feature_engineer import FeatureEngineer

def main():
    # Caminhos
    clean_data_path = project_root / "data" / "processed" / "dataset_clean.csv"
    
    print("=" * 60)
    print("  Treinamento do Feature Engineer (Escalas e Categorias)")
    print("=" * 60)

    if not clean_data_path.exists():
        print(f"[ERRO] Dataset limpo não encontrado em: {clean_data_path}")
        print("DICA: Rode 'uv run python scripts/clean_dataset.py' primeiro.")
        return

    # 1. Carregar dados reais
    print(f"\n[1/3] Carregando dados de: {clean_data_path}")
    df = pd.read_csv(clean_data_path)

    # 2. Definir features
    # Aqui o aluno escolhe o que quer que a IA aprenda
    numeric_features = ["tempo", "popularity", "danceability", "energy"]
    categorical_features = ["track_genre"]

    # 3. Fit (Aprendizado)
    print(f"\n[2/3] Iniciando o FIT (Aprendizado de {len(df)} linhas)...")
    fe = FeatureEngineer()
    fe.fit(df, numeric_features, categorical_features)

    # 4. Transformação e Salvamento do Dataset de Atributos
    print(f"\n[3/4] Gerando dataset transformado (Matriz de Atributos)...")
    df_transformed = fe.transform(df)
    
    output_path = project_root / "data" / "processed" / "dataset_features.csv"
    df_transformed.to_csv(output_path, index=False)

    # 5. Verificação
    print("\n[4/4] Sucesso! Tudo pronto para a Semana 06.")
    print(f"      - Transformadores salvos em: models/transformers.joblib")
    print(f"      - Dataset de Atributos salvo em: {output_path}")
    print(f"      - Total de colunas (Features): {len(fe.feature_names)}")

if __name__ == "__main__":
    main()