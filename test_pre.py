import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv('survey_lung_cancer.csv')

# 2. การเตรียมข้อมูล
df['AGE'] = (df['AGE'] - df['AGE'].min()) / (df['AGE'].max() - df['AGE'].min())
X = df[['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ','ALLERGY ','WHEEZING','ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN']]
Y = df['LUNG_CANCER']

# Normalize X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# เพิ่ม bias term
X = np.c_[np.ones(X.shape[0]), X]

# การกำหนดค่าพารามิเตอร์
w = np.zeros(X.shape[1])
alpha = 0.0001
epochs = 10000
n = len(Y)

# การฝึกโมเดลด้วย Gradient Descent
costs = []

for i in range(epochs):
    Y_hat = np.dot(X, w)
    mse = (1 / (2 * n)) * np.sum((Y_hat - Y) ** 2)
    costs.append(mse)
    gradient = np.dot(X.T, (Y_hat - Y)) / n
    w -= alpha * gradient

    if i % 1000 == 0:
        print(f"MSE {i}: {mse}")


Y_prob = 1 / (1 + np.exp(-np.dot(X, w)))

# 8. แสดงผลลัพธ์ในรูปแบบเปอร์เซ็นต์
percentages = Y_prob * 100

num_samples_to_show = 5

# เลือกตัวอย่างสุ่ม
indices_to_show = np.random.choice(len(percentages), num_samples_to_show, replace=False)

# แสดงผลลัพธ์ของตัวอย่างสุ่ม
for idx in indices_to_show:
    percent = percentages[idx]
    print(f"Probability person {idx+1}: {percent:.2f}%")

print("Final weights:", w)

# Plot Loss over Epochs
plt.plot(range(epochs), costs)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# Plot Predictions vs Actuals
plt.scatter(range(len(Y)), Y, color='blue', label='Truth')
plt.scatter(range(len(Y)), Y_hat, color='red', label='Predict the probability of that person')
plt.title("Predictions vs Truth")
plt.xlabel("Sample person")
plt.ylabel("Lung Cancer (0=No, 1=Yes)")
plt.legend()
plt.show()
