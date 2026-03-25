import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureEngineer:
    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.transformer_path = self.model_dir / "transformers.joblib"
        self.pipeline = None
        self.feature_names = None
    
    def _build_pipeline(self, numeric_features: list[str], categorical_features: list[str]):
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", MinMaxScaler(), numeric_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
                ]
            )
            return Pipeline(steps=[("preprocessor", preprocessor)])
    
    def fit(self, df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]):
            """Aprende as escalas e categorias dos dados e salva o pipeline."""
            self.pipeline = self._build_pipeline(numeric_features, categorical_features)
            self.pipeline.fit(df)
            cat_encoder = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"]
            cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
            self.feature_names = numeric_features + cat_features
            self.save()
            return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            """Aplica a transformação salva em novos dados."""
            if self.pipeline is None:
                self.load()
            transformed_data = self.pipeline.transform(df)
            return pd.DataFrame(transformed_data, columns=self.feature_names)
        
    def save(self):
            joblib.dump({"pipeline": self.pipeline, "feature_names": self.feature_names}, self.transformer_path)

    def load(self):
            state = joblib.load(self.transformer_path)
            self.pipeline = state["pipeline"]
            self.feature_names = state["feature_names"]

        