import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

# Leer el archivo CSV y almacenarlo en un DataFrame
data = pd.read_csv('framingham.csv')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Crear un objeto SimpleImputer para reemplazar NaN con la media de la columna en columnas numéricas
num_imputer = SimpleImputer(strategy='mean')

# Aplicar la imputación a las columnas numéricas
data[['cigsPerDay', 'BPMeds', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']] = num_imputer.fit_transform(data[['cigsPerDay', 'BPMeds', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']])

# Crear un objeto SimpleImputer para reemplazar NaN con la moda en columnas categóricas
cat_imputer = SimpleImputer(strategy='most_frequent')

# Aplicar la imputación a la columna 'education'
data['education'] = cat_imputer.fit_transform(data['education'].values.reshape(-1, 1))

# Convertir el DataFrame en un np.array
data_array = data.to_numpy()

# Separar las características y la variable objetivo
X = data_array[:, :-1]
y = data_array[:, -1]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Aplicar transformación polinomial
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)) 
        cost_history[i] = cost_function(X, y, params)

    return (cost_history, params)

def logistic_regression(X, y, learning_rate, iterations):
    initial_params = np.zeros((X.shape[1], 1))
    cost_history, params_optimal = gradient_descent(X, y, initial_params, learning_rate, iterations)
    return params_optimal


#task1.4

# Determinar el grado del polinomio que mejor describe la nube de puntos
max_degree = 5
best_degree = 1
best_score = 0

for degree in range(1, max_degree + 1):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    logistic_model = LogisticRegression()
    scores = cross_val_score(logistic_model, X_poly, y, cv=5)
    
    if np.mean(scores) > best_score:
        best_score = np.mean(scores)
        best_degree = degree

print(f"El mejor grado del polinomio es: {best_degree}")

# Crear el modelo logístico polinomial con grado 1 porque ese encontramos que es el mejor
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
logistic_model = LogisticRegression()
logistic_model.fit(X_train_poly, y_train)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy = logistic_model.score(X_test_poly, y_test)
print("Precisión del modelo en el conjunto de prueba:", accuracy)
