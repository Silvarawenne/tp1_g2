"""Modelos clássicos (não profundos) e seus Pipelines completos.

Requisito central (enunciado §4.4, erro comum #2): normalizador, PCA/seleção e
SMOTE devem estar DENTRO do Pipeline, para serem ajustados somente na partição de
treino de cada dobra — nunca no conjunto completo antes da divisão.

Todos os modelos herdam `random_state=SEED` para reprodutibilidade (enunciado §4.5).
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import SEED


class SafeSMOTE(SMOTE):
    """SMOTE que reduz automaticamente `k_neighbors` quando a classe minoritária da
    dobra de treino tem poucos exemplos, em vez de lançar erro.

    Necessário porque, ao particionar por paciente (StratifiedGroupKFold) com poucos
    positivos, uma dobra de treino pode conter menos exemplos minoritários do que o
    k_neighbors=5 padrão do SMOTE — situação comum tanto em amostras estratificadas
    pequenas quanto no smoke-test com dados sintéticos.
    """

    def _fit_resample(self, X, y):
        counts = np.bincount(y)
        minority_count = int(counts[counts > 0].min())
        if minority_count <= 1:
            # Não há vizinhos suficientes para interpolar — não reamostra nesta dobra.
            return X, y
        self.k_neighbors = max(1, min(5, minority_count - 1))
        return super()._fit_resample(X, y)

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False


def build_trivial_baseline() -> DummyClassifier:
    """Classificador de classe majoritária — obrigatório (enunciado §4.3) para
    interpretar as demais métricas em uma base desbalanceada.
    """
    return DummyClassifier(strategy="most_frequent", random_state=SEED)


def _with_preprocessing(estimator, use_pca: bool = False, use_smote: bool = True, n_features_select: int | None = None) -> ImbPipeline:
    steps = [("scaler", StandardScaler())]
    if n_features_select is not None:
        steps.append(("select", SelectKBest(score_func=f_classif, k=n_features_select)))
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=SEED)))
    if use_smote:
        steps.append(("smote", SafeSMOTE(random_state=SEED)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def build_model_zoo(use_smote: bool = True) -> dict[str, ImbPipeline]:
    """Retorna o dicionário {nome: pipeline} com >= 3 famílias de modelos clássicos
    exigidas pelo enunciado §4.3: SVM (linear e RBF), Random Forest, gradient boosting,
    k-NN, regressão logística regularizada e MLP raso.
    """
    zoo = {
        "logreg": _with_preprocessing(
            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED),
            use_smote=use_smote,
        ),
        "svm_linear": _with_preprocessing(
            SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=SEED),
            use_smote=use_smote,
        ),
        "svm_rbf": _with_preprocessing(
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED),
            use_smote=use_smote,
        ),
        "knn": _with_preprocessing(
            KNeighborsClassifier(n_neighbors=5, weights="distance"),
            use_smote=use_smote,
        ),
        "random_forest": _with_preprocessing(
            RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=SEED, n_jobs=-1),
            use_smote=use_smote,
        ),
        "mlp_raso": _with_preprocessing(
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, early_stopping=True, random_state=SEED),
            use_smote=use_smote,
        ),
    }

    if _HAS_XGBOOST:
        zoo["xgboost"] = _with_preprocessing(
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
            ),
            use_smote=use_smote,
        )
    else:  # pragma: no cover - fallback caso xgboost não esteja instalado
        zoo["gradient_boosting"] = _with_preprocessing(
            GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED),
            use_smote=use_smote,
        )

    return zoo


PARAM_GRIDS = {
    "logreg": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "svm_linear": {"clf__C": [0.1, 1.0, 10.0]},
    "svm_rbf": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]},
    "knn": {"clf__n_neighbors": [3, 5, 9, 15]},
    "random_forest": {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [None, 6, 12]},
    "mlp_raso": {"clf__hidden_layer_sizes": [(16,), (32,), (64,)], "clf__alpha": [1e-4, 1e-3, 1e-2]},
    "xgboost": {"clf__max_depth": [3, 4, 6], "clf__learning_rate": [0.01, 0.05, 0.1]},
    "gradient_boosting": {"clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.01, 0.05, 0.1]},
}
