import os.path

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.data_io import DataReader, DataWriter
from analysis.preprocessing import DataCleaning
from utils.consts import DEFAULT_SCORING, NA_THRESHOLD, OUTPUT_LOG_FILE, OUTPUT_PATH, RANDOM_STATE, KFOLD_SPLITS
from utils.helpers import dprint


class Analysis:
    def __init__(self, path, study, debug, **kwargs):
        self.path = os.path.join(path, kwargs.get("dir"))
        self.study = study
        self.debug = debug

        self.model = kwargs.get('classifier')
        self.preprocessor = kwargs.get('preprocessor')
        self.preproc_kwargs = kwargs.get('preproc_kwargs')
        self.classifier_param_grid = kwargs.get('classifier_param_grid')

        self._log_text = ""

    def run(self):
        self.print(f'Running analysis for study {self.study}')

        genera, metadata, mtb = DataReader.read_files(self.path)
        genera, metadata, mtb = DataCleaning.clean_data(genera, metadata, mtb, na_threshold=NA_THRESHOLD)

        self.log(f"Genera shape: {genera.shape}")
        self.log(f"Metadata shape: {metadata.shape}")
        self.log(f"MTB shape: {mtb.shape}")

        self.log(f"Group distribution: {metadata['Study.Group'].value_counts()}")

        # Genera
        features = self.preprocessor().fit_transform(genera[:, 1:], metadata["Study.Group"], **self.preproc_kwargs)

        kfold = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv = GridSearchCV(estimator=self.model(), param_grid=self.classifier_param_grid, cv=kfold,
                          scoring=DEFAULT_SCORING) \
            .fit(features, metadata["Study.Group"])

        self.log(f"Best hyperparameters: {cv.best_params_}")
        self.log(f"Best score: {cv.best_score_}")

        best_model = cv.best_estimator_
        y_pred = best_model.predict(features)
        self.log(f'Best model: {best_model}')

        self.visualize_embeddings(genera, metadata, mtb)
        self.evaluate_performance(metadata["Study.Group"], y_pred, metadata)

    def evaluate_performance(self, y_true, y_pred, metadata):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        confusion_matrix = confusion_matrix(y_true, y_pred)
        classification_report = classification_report(y_true, y_pred)
        roc_curve = roc_curve(y_true, y_pred)
        precision_recall_curve = precision_recall_curve(y_true, y_pred)

        self.log(f"Accuracy: {accuracy}")
        self.log(f"Precision: {precision}")
        self.log(f"Recall: {recall}")
        self.log(f"F1: {f1}")
        self.log(f"ROC AUC: {roc_auc}")
        self.log(f"Confusion matrix: {confusion_matrix}")
        self.log(f"Classification report: {classification_report}")
        self.log(f"ROC curve: {roc_curve}")
        self.log(f"Precision-recall curve: {precision_recall_curve}")

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'{OUTPUT_PATH}{self.study}-roc.pdf')

        plt.plot(precision_recall_curve[1], precision_recall_curve[0], marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(f'{OUTPUT_PATH}{self.study}-prc.pdf')

        DataWriter.write_files(OUTPUT_PATH, f"{self.study}-{OUTPUT_LOG_FILE}", self._log_text)

    def visualize_embeddings(self, genera, metadata, mtb):
        fig, ax = plt.subplots(1, 3, figsize=(14, 3), gridspec_kw={'wspace': 0.25})
        sns.scatterplot(data=metadata, x="BMI", y="Study.Group", ax=ax[0], hue="Study.Group")
        sns.kdeplot(data=metadata, x="Age", hue="Study.Group", ax=ax[1])
        metadata.groupby(['Gender', 'Study.Group']).size().unstack().plot(kind='bar', stacked=True, ax=ax[2])
        ax[2].set(xlabel="", ylabel="Count")
        plt.suptitle("Subjects' metadata")
        plt.savefig(f'{OUTPUT_PATH}{self.study}-metadata.pdf')

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        tsne = TSNE(n_components=2, random_state=42)
        mtb_tsne = tsne.fit_transform(mtb.values[:, 1:])
        tsne_df = pd.DataFrame(data=mtb_tsne, columns=['TSNE1', 'TSNE2'])
        tsne_df['Study.Group'] = metadata['Study.Group']
        sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Study.Group', ax=ax[0])
        ax[0].set_title('TSNE on MTB')
        tsne = TSNE(n_components=2, random_state=42)
        gen_tsne = tsne.fit_transform(genera.values[:, 1:])
        tsne_df = pd.DataFrame(data=gen_tsne, columns=['TSNE1', 'TSNE2'])
        tsne_df['Study.Group'] = metadata['Study.Group']
        sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Study.Group', ax=ax[1])
        ax[1].set_title('TSNE on Genera')
        plt.suptitle("TSNE on MTB and Genera")
        plt.savefig(f'{OUTPUT_PATH}{self.study}-tsne.pdf')

    def log(self, msg):
        self._log_text += msg + '\n'
        self.print(msg)

    def print(self, msg):
        dprint(msg, self.debug)
