import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO

# Colunas numéricas do dataset Spotify que usaremos para análise
SPOTIFY_NUMERIC_COLS = [
    "popularity", "duration_ms", "danceability", "energy",
    "loudness", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

# Colunas obrigatórias para o upload de biblioteca
REQUIRED_LIBRARY_COLS = ["track_name", "artists", "energy", "loudness"]

class DataCleaner:
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "DataCleaner":
        """Carrega dados de um arquivo CSV no disco."""
        df = pd.read_csv(path, **kwargs)
        return cls(df=df)

    @classmethod
    def from_bytes(cls, content: bytes, file_type: str = "csv") -> "DataCleaner":
        """Carrega dados a partir de bytes (usado pelo endpoint de upload)."""
        buffer = BytesIO(content)
        if file_type == "json":
            df = pd.read_json(buffer)
        else:
            df = pd.read_csv(buffer)
        return cls(df=df)
    
    def diagnose(self) -> dict:
        df = self.df
        report = {}

        report["total_rows"] = int(len(df))
        report["total_columns"] = int(len(df.columns))
        report["duplicate_rows"] = int(df.duplicated().sum())

        # Relatório por coluna
        columns_report = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            total = len(df)
            sample_values = df[col].dropna().head(3).tolist()
            sample_values = [v.item() if hasattr(v, "item") else v for v in sample_values]

            columns_report.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_pct": round(missing / total * 100, 2) if total > 0 else 0,
                "unique_count": int(df[col].nunique()),
                "sample_values": sample_values,
            })
        report["columns"] = columns_report

        # Detecção de Outliers (IQR)
        outliers_report = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            total_notna = int(df[col].notna().sum())

            outliers_report.append({
                "column": col,
                "total_outliers": n_outliers,
                "outlier_pct": round(n_outliers / total_notna * 100, 2) if total_notna > 0 else 0,
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
            })
        report["outliers"] = outliers_report

        # Estatísticas e correlações
        if numeric_cols:
            report["numeric_summary"] = df[numeric_cols].describe().round(4).to_dict()
            if len(numeric_cols) >= 2:
                report["correlations"] = df[numeric_cols].corr().round(4).to_dict()
            else:
                report["correlations"] = {}
        else:
            report["numeric_summary"] = {}
            report["correlations"] = {}

        report["health_score"] = self._calculate_health_score(report)
        return report
    
    def clean(self, save_path: str | Path | None = None) -> pd.DataFrame:
        df = self.df.copy()

        # 1. Remover coluna de índice duplicado do CSV original
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # 2. Remover duplicatas
        if "track_id" in df.columns:
            df = df.drop_duplicates(subset=["track_id"], keep="first")
        else:
            df = df.drop_duplicates(keep="first")

        # 3. Tratar valores nulos
        text_critical = [c for c in ["track_name", "artists"] if c in df.columns]
        if text_critical:
            df = df.dropna(subset=text_critical)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # 4. Normalizar loudness (escala 0-1)
        if "loudness" in df.columns:
            loud_min = df["loudness"].min()
            loud_max = df["loudness"].max()
            if loud_max != loud_min:
                df["loudness_norm"] = ((df["loudness"] - loud_min) / (loud_max - loud_min)).round(6)
            else:
                df["loudness_norm"] = 0.0

        # 5. Salvar (opcional)
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)

        self.df = df
        return df
    
    def _calculate_health_score(self, report: dict) -> float:
        total_rows = report["total_rows"]
        total_cols = report["total_columns"]

        if total_rows == 0 or total_cols == 0:
            return 0.0

        # Completude (40 pts)
        total_missing = sum(c["missing_count"] for c in report["columns"])
        total_cells = total_rows * total_cols
        completude = (1 - total_missing / total_cells) * 40

        # Unicidade (30 pts)
        dup = report["duplicate_rows"]
        unicidade = (1 - dup / total_rows) * 30

        # Consistência / Outliers (30 pts)
        if report["outliers"]:
            total_outlier_cells = sum(o["total_outliers"] for o in report["outliers"])
            total_numeric_cells = sum(total_rows for _ in report["outliers"])
            consistencia = (1 - total_outlier_cells / total_numeric_cells) * 30
        else:
            consistencia = 30.0

        score = completude + unicidade + consistencia
        return round(max(0, min(100, score)), 1)
    
    def validate_library_upload(self) -> dict:
        df = self.df
        result = {
            "total_received": len(df), "total_valid": 0, "total_invalid": 0,
            "invalid_rows": [], "sample": []
        }

        # Verificar colunas obrigatórias
        missing_cols = [c for c in REQUIRED_LIBRARY_COLS if c not in df.columns]
        if missing_cols:
            result["invalid_rows"].append({"row": -1, "reason": f"Colunas ausentes: {missing_cols}"})
            result["total_invalid"] = len(df)
            return result

        mask_valid = pd.Series(True, index=df.index)

        # Verificar nulos em colunas obrigatórias
        for col in REQUIRED_LIBRARY_COLS:
            null_mask = df[col].isna()
            if null_mask.any():
                for idx in df[null_mask].index:
                    result["invalid_rows"].append({"row": int(idx), "reason": f"Nulo em '{col}'"})
                mask_valid = mask_valid & ~null_mask

        # Verificar tipos numéricos
        for col in ["energy", "loudness"]:
            if col in df.columns:
                numeric_mask = pd.to_numeric(df[col], errors="coerce").notna()
                invalid = ~numeric_mask & df[col].notna()
                if invalid.any():
                    for idx in df[invalid].index:
                        result["invalid_rows"].append({"row": int(idx), "reason": f"Não numérico em '{col}'"})
                    mask_valid = mask_valid & numeric_mask

        result["total_valid"] = int(mask_valid.sum())
        result["total_invalid"] = int((~mask_valid).sum())

        valid_df = df[mask_valid].head(5)
        sample = valid_df.to_dict(orient="records")
        for record in sample:
            for k, v in record.items():
                if hasattr(v, "item"):
                    record[k] = v.item()
        result["sample"] = sample

        return result
    