# ANA680 Final Project — Wine Quality Prediction

This repository is the starter structure for the ANA680 Final Project.

## Project Objective
Develop an end-to-end machine learning system that predicts red wine quality from physicochemical measurements and demonstrate the model through Flask, Docker, CI/CD/Heroku, AWS SageMaker, and Kubernetes.

## Dataset
UCI Wine Quality — Red Wine dataset.

> Add `winequality-red.csv` to the `data/` folder before running the notebook or training script.

## Project Components
1. Problem definition and dataset
2. EDA, cleaning, feature engineering/selection
3. Train/validation/test split, model comparison, evaluation, and `.pkl` model
4. Local Flask deployment
5. Docker container and Docker Hub
6. GitHub Actions CI/CD and Heroku
7. AWS SageMaker Studio using containers
8. Kubernetes deployment
9. Recorded demonstration

## Local Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Docker
```bash
docker build -t wine-quality-final .
docker run -p 5000:5000 wine-quality-final
```

## Kubernetes
Update the image name in `kubernetes/deployment.yaml`, then:
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods
kubectl get services
```

## Important
Replace placeholder values in the deployment files with your actual Docker Hub, Heroku, and AWS information before submission.


## Final Model Results

The included training workflow was executed with the UCI Red Wine Quality dataset.

- **Best model:** Random Forest
- **Test MAE:** 0.4600
- **Test RMSE:** 0.5950
- **Test R²:** 0.4375
- **Rows after cleaning:** 1359
- **Data split:** 70% training / 15% validation / 15% testing

The trained model is included as `wine_quality_model.pkl`, and detailed results are saved in `model_metrics.json`.