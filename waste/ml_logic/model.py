from waste.params import *

from comet_ml import API
import logging


def load_best_weights():
    api = API()

    # Fetching the model from Comet ML

    models = api.get_model(
        workspace=COMET_WORKSPACE_NAME,
        model_name=COMET_MODEL_NAME,
    )

    # Get production model weights
    model_versions = models.find_versions(status="Production")
    latest_production_weights = model_versions[0]

    # Downloading the weights
    models.download(
        version=latest_production_weights,
        output_folder=WEIGHTS_PATH,
        expand=True,
    )
    logging.info("✅ Loaded weights from the comet ML")

    return os.path.join(WEIGHTS_PATH, "best.pt")


def update_production_model():
    api = API()

    # Save the trained model weights to Comet ML
    experiments = api.get(
        workspace=COMET_WORKSPACE_NAME, project_name=COMET_PROJECT_NAME
    )

    # Registering the latest experiment and model
    current_experiment = experiments[-1]._name
    experiment = api.get(
        workspace=COMET_WORKSPACE_NAME,
        project_name=COMET_PROJECT_NAME,
        experiment=current_experiment,
    )

    # Sort list of experiments by one of the metrics to find best one
    experiments.sort(
        key=lambda each_experiment: float(
            each_experiment.get_metrics_summary("metrics/mAP50(B)")["valueMax"]
        )
        # If some experiment got stopped without any metric we want to skip it
        if isinstance(each_experiment.get_metrics_summary("metrics/mAP50(B)"), dict)
        else 0
    )

    # get best experiment
    best_experiment_so_far = experiments[-1]._name

    # If current one is the best than move this model to production
    if current_experiment == best_experiment_so_far:
        experiment.register_model(COMET_MODEL_NAME, status="Production")
        logging.info("✅ Registered current model as Production")
    else:
        experiment.register_model(COMET_MODEL_NAME)
        logging.info("🖌️ Registered current model as history")
