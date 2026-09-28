import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=100, random_state=42):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        """
        X: ma trận (n_samples, n_features)
        y: nhãn, dùng {-1, +1}
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        # khởi tạo w (chưa gồm bias) và bias riêng
        self.w = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b = 0.0
        self.errors_per_epoch = []

        for epoch in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_pred = self._predict_raw(xi)
                if y_pred != yi:  # phân lớp sai -> cập nhật
                    update = self.lr * yi
                    self.w += update * xi
                    self.b += update
                    errors += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:   # hội tụ, dừng sớm
                break
        return self

    def _net_input(self, xi):
        return np.dot(xi, self.w) + self.b

    def _predict_raw(self, xi):
        return 1 if self._net_input(xi) >= 0 else -1

    def predict(self, X):
        return np.array([self._predict_raw(xi) for xi in X])
X = np.array([[3,4],[2,3],[1,1],[0,0]])
y = np.array([1,-1,-1,1])
model = Perceptron(learning_rate=0.1, n_epochs=50)
model.fit(X, y)
print(model.predict(X))