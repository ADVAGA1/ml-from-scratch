class SimpleLinearRegression():
    def __init__(self, iterations: int, learning_rate: float):
        self.iterations = iterations
        self.learning_rate = learning_rate
        
        self.slope: float = 0
        self.intercept: float = 0
        

    def _mse(self, y_hat: list[float], y: list[float]) -> float:
        sum = 0

        for i in range(len(y)):
            value = y[i] - y_hat[i]
            value = value ** 2
            sum += value
        
        return sum / len(y)
    
    def _mae(self, y_hat: list[float], y: list[float]) -> float:
        sum = 0
        for i in range(len(y)):
            value = y_hat[i] - y[i]
            sum += abs(value)
        
        return sum / len(y)
    
    def r_squared(self, y_hat: list[float], y: list[float]) -> float:
        ss_res = sum([(y[i] - y_hat[i]) ** 2 for i in range(len(y))])
        y_mean = sum(y) / len(y)
        ss_tot = sum([(y[i] - y_mean) ** 2 for i in range(len(y))])

        return 1 - ss_res / ss_tot

    def _get_slope_gradient(self, X: list[float], y:list[float], y_hat: list[float]) -> float:
        sum = 0
        for i in range(len(X)):
            value = X[i] * (y[i] - y_hat[i])
            sum += value
        
        return -2 * sum / len(X)
    
    def _get_intercept_gradient(self, y: list[float], y_hat: list[float]) -> float:
        sum = 0
        for i in range(len(y)):
            value = y[i] - y_hat[i]
            sum += value

        return -2 * sum / len(y)

    def predict(self, X: list[float]) -> list[float]:
        return [x_i * self.slope + self.intercept for x_i in X]

    def fit(self, X: list[float], y: list[float]) -> None:
        
        for _ in range(self.iterations):
            
            predictions = self.predict(X)

            # mse = self._mse(predictions, y)

            slope_gradient = self._get_slope_gradient(X, y, predictions)
            intercept_gradient = self._get_intercept_gradient(y, predictions)

            self.slope = self.slope - self.learning_rate * slope_gradient
            self.intercept = self.intercept - self.learning_rate * intercept_gradient

    def eval(self, y_hat: list[float], y: list[float]) -> None:
        mse = self._mse(y_hat, y)
        mae = self._mae(y_hat, y)
        r_squared = self.r_squared(y_hat, y)

        print("Model evaluation:")
        print("####################")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2: {r_squared:.4f}")

