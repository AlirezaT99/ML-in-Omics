import os

from utils.helpers import dprint


class Analysis:
    def __init__(self, path, study, debug, **kwargs):
        self.data_path = os.path.join(path, study)
        self.debug = debug

        self.model = kwargs.get('classifier')
        self.preprocessor = kwargs.get('preprocessor')
        self.preproc_kwargs = kwargs.get('preproc_kwargs')
        self.classifier_param_grid = kwargs.get('classifier_param_grid')

    def run(self):
        self.print(f'Running analysis for study {self.study} at path {self.path}')

        # self._data = self.reader.read_data(os.path.join(DATA_PATH, self.path), **self.reader_kwargs)
        # self._embedding = self.preprocessor.get_embedding(self._data.seq, **self.preproc_kwargs)

        # features = self.preprocessor.prepare_features(self._embedding)
        # self.classifier.train_model(features, self._data.label, self.clf_params)
        # y_pred = self.classifier.predict(features)

        # visualize_embeddings(features, self._data.label, self.name)
        # evaluate_performance(y_true=self._data.label, y_pred=y_pred, metrics=[METRICS[0], METRICS[6]], name=self.name)

    def print(self, msg):
        dprint(msg, self.debug)