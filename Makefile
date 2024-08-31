install:
	@pip install -e .

train:
	@python waste/interface/main.py

run_api:
	@uvicorn waste.api.fast:app --reload --port 8080

build_training_mac:
	@docker build --platform linux/amd64 --file Dockerfile.training -t poloniki/${IMAGE_NAME}_training:latest .
push_training:
	@docker push poloniki/${IMAGE_NAME}_training:latest

build_api_mac:
	@docker build --platform linux/amd64 --file Dockerfile.training -t ${ARTIFACT_PATH}/${IMAGE_NAME}_api:latest .
push_api:
	@docker push ${ARTIFACT_PATH}/${IMAGE_NAME}_api:latest
deploy_api:
	gcloud run deploy --image ${ARTIFACT_PATH}/${IMAGE_NAME}_api:latest --memory 8Gib --region europe-west1  --env-vars-file .env.yaml
