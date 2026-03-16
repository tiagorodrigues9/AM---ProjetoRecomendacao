import sys
from pathlib import Path

# Adiciona a raiz do projeto no path do script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.data_cleaner import DataCleaner

def main():
    raw_path = project_root / "data" / "raw" / "dataset.csv"
    clean_path = project_root / "data" / "processed" / "dataset_clean.csv"

    print("=" * 60)
    print("  Pipeline de Limpeza - Dataset Spotify")
    print("=" * 60)

    # 1. Carregar
    print(f"\n[1/3] Carregando dados de: {raw_path}")
    cleaner = DataCleaner.from_csv(raw_path)

    # 2. Diagnóstico Prévio
    print("\n[2/3] Diagnóstico ANTES da limpeza:")
    report = cleaner.diagnose()
    print(f"      Health Score: {report['health_score']}/100")

    # 3. Limpar e salvar
    print("\n[3/3] Executando pipeline de limpeza...")
    cleaner.clean(save_path=clean_path)

    # 4. Diagnóstico final
    print("\n--- Diagnóstico DEPOIS da limpeza ---")
    report_after = cleaner.diagnose()
    print(f"      Health Score: {report_after['health_score']}/100")

if __name__ == "__main__":
    main()