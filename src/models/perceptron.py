class Perceptron:
    def __init__(self, weights=None, bias=0.1):
        default_weights = {'energy': 0.8, 'loudness': 0.2}
        self.weights = weights if weights is not None else default_weights
        self.bias = bias

    def predict(self, energy, loudness):
        # Normalização: (-60dB a 0dB) -> (aprox 0.0 a 1.0)
        # A ideia é trazer o loudness (negativo) para uma escala comparável
        loudness_norm = (loudness + 10) / 10

        # Cálculo Z
        w_energy = self.weights.get('energy', 0.0)
        w_loudness = self.weights.get('loudness', 0.0)
        linear_output = (energy * w_energy) + (loudness_norm * w_loudness) + self.bias

        # Ativação (Degrau)
        prediction = 1 if linear_output >= 0.5 else 0

        return {
            "prediction": prediction,
            "activation": linear_output,
            "normalized_loudness": loudness_norm
        }