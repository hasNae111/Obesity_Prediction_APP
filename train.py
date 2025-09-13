import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib

# === 1. Charger les données ===
df = pd.read_csv("data.csv")  # Remplace par le chemin réel

# === 2. Séparer features et cible ===
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# === 3. Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Encodage automatique des colonnes textuelles ===
# Convertir toutes les colonnes object restantes en colonnes numériques
object_cols = X_train.select_dtypes(include='object').columns.tolist()

X_train = pd.get_dummies(X_train, columns=object_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=object_cols, drop_first=True)

# Aligner les colonnes train/test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# === 5. Normalisation pour modèles linéaires / SVM ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. Définition des modèles ===
models = {
    "Logistic Regression": LogisticRegression(multi_class="multinomial", max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

# === 7. Entraînement et évaluation ===
results = []
for name, model in models.items():
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    results.append({"Modèle": name, "Accuracy": acc, "F1-macro": f1})

# === 8. Affichage des résultats ===
results_df = pd.DataFrame(results)
print("Résultats des modèles :")
print(results_df)

# === 9. Sauvegarde du meilleur modèle ===
best_result = max(results, key=lambda x: x["F1-macro"])
best_model_name = best_result["Modèle"]
best_model = models[best_model_name]
print(f"Meilleur modèle : {best_model_name}")

# Réentraîner le meilleur modèle sur toutes les données
object_cols_full = X.select_dtypes(include='object').columns.tolist()
X_full = pd.get_dummies(X, columns=object_cols_full, drop_first=True)
X_full = X_full.reindex(columns=X_train.columns, fill_value=0)

if best_model_name in ["Logistic Regression", "SVM"]:
    best_model.fit(scaler.fit_transform(X_full), y)
    joblib.dump(scaler, "scaler.pkl")
else:
    best_model.fit(X_full, y)

joblib.dump(best_model, "model.pkl")
print("Modèle sauvegardé sous model.pkl")
