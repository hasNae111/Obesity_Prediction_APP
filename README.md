# Obesity Prediction App

Web app pour prédire le risque d'obésité avec FastAPI, PostgreSQL et ML.

## Installation

1. Cloner le projet :

```bash
git clone https://github.com/votre-utilisateur/obesity-prediction-app.git
cd obesity-prediction-app
```

2. Installer dépendances :

```bash
pip install -r requirements.txt
```

3. Copier `.env.example` → `.env` et remplir.
4. Lancer entraînement :

```bash
python app/ml/train.py
```

5. Démarrer l'API :

```bash
uvicorn app.main:app --reload
```

Accéder à l'API : `http://localhost:8000/docs`

## Docker

```bash
docker compose up --build
```

## Endpoints principaux

* `/auth/register` → Inscription
* `/auth/login` → Connexion
* `/predict/` → Prédiction
* `/history/` → Historique
* `/admin/users` → Admin
* `/metrics/` → Infos modèle
