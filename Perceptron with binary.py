import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv("creditcard.csv")
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=6,
    n_classes=2, weights=[0.8, 0.2], random_state=42
)
y = np.where(y == 0, -1, 1)  # Perceptron cần nhãn {-1, +1}

# 2. Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 3. Chuẩn hóa dữ liệu (quan trọng với Perceptron)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Huấn luyện mô hình
model = Perceptron(eta0=0.01, max_iter=200)
model.fit(X_train, y_train)

# 5. Dự báo và đánh giá
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
