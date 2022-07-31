from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from datetime import datetime


GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "./data/news_classifier.joblib"
        }
    },
    "service": {
        "log_destination": "./data/logs.out"
    }
}

class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    #estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    #transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        self.config = config

        sentence_transformer_model = SentenceTransformer(
            self.config['featurizer']['sentence_transformer_model']
        )
        featurizer = TransformerFeaturizer(
            dim=self.config['featurizer']['sentence_transformer_embedding_dim'],
            sentence_transformer_model=sentence_transformer_model
        )

        model = joblib.load(self.config['classifier']['serialized_model_path'])

        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', model)
        ])

        self.classes = self.pipeline.classes_

    def predict_proba(self, model_input: dict) -> dict:
        output = self.pipeline.predict_proba(
            [model_input['description']]
        )[0]

        return {label: score for label, score in zip(self.classes, output)}

    def predict_label(self, model_input: dict) -> str:
        output = self.pipeline.predict(
            [model_input['description']]
        )[0]

        return output


app = FastAPI()
classifier = None

@app.on_event("startup")
def startup_event():
    global classifier
    classifier = NewsCategoryClassifier(config=GLOBAL_CONFIG['model'])

    logger.add(GLOBAL_CONFIG['service']['log_destination'])
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    logger.info("Shutting down application")
    logger.complete()
    logger.remove()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    t0 = datetime.utcnow()

    request_dict = request.dict()

    scores = classifier.predict_proba(request_dict)
    label = classifier.predict_label(request_dict)

    response = PredictResponse(
        scores=scores,
        label=label
    )

    t1 = datetime.utcnow()

    logger.info({
        'timestamp': t0.strftime('%Y-%m-%d %H:%M:%S'),
        'request': request_dict,
        'prediction': response.dict(),
        'latency': round((t1 - t0).total_seconds() * 1000)
    })

    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
