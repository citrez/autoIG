models_to_deploy = [
    # Try add another to this
    {
        "model_name": "knn-reg-model",
        "model_version": 32,
        "r_threshold": 0.9, # Should this be set here, or in mlflow for each model? Surely this should be included in trainnig, and is not a deployment charichteristic
    }
]
