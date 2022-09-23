import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC

from learning_models.logistic_regression.logistic_regression import LogisticRegressionGroupedSamples
from library.explainability import _return_perm_imp, _plot_shap
from library.utils.print_functions import ColorPrint


def train_learning_model(learning_model_type, perm_n_repeats, plot_SHAPLEY):

    learning_model_functions = {
        'Logistic Regression': LogisticRegression(n_jobs=-1, max_iter=500),
        'Logistic Regression Grouped Samples': LogisticRegressionGroupedSamples(n_jobs=-1, max_iter=500,
                                                                                tol=1e-6),
        'Logistic Regression CV': LogisticRegressionCV(n_jobs=-1, max_iter=500),
        # 'Ridge Classifier': RidgeClassifier(),
        # 'Ridge Classifier CV': RidgeClassifierCV(),
        'Linear SVC': LinearSVC(),
        'SVC': SVC(probability=True),
        'NuSVC': NuSVC(probability=True, nu=0.01),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(max_features=2),
        'AdaBoost': AdaBoostClassifier(),
        'MLP': MLPClassifier(hidden_layer_sizes=(1,), max_iter=1000)
    }

    def _train_model(features_df, sel_columns, sample_weight=None):
        ColorPrint("Training {}".format(learning_model_type), 'OKBLUE')
        print("DEBUG _train_model:, sel_columns=", sel_columns)

        importances_df = pd.DataFrame([])

        if learning_model_type == 'Logistic Regression Grouped Samples':
            learning_model_functions[learning_model_type] \
                .fit(X=features_df[sel_columns], y=features_df['is_active'],
                     X_class=features_df['protein'].factorize()[0],
                     sample_weight=sample_weight)
        else:
            learning_model_functions[learning_model_type] \
                .fit(X=features_df[sel_columns], y=features_df['is_active'])

        if learning_model_type.startswith('Logistic Regression'):
            print('Coefficients:', learning_model_functions[learning_model_type].coef_,
                  learning_model_functions[learning_model_type].intercept_)
        elif learning_model_type in ['Gradient Boosting', 'Random Forest', 'AdaBoost']:
            importances_df = pd.DataFrame({f: [i] for f,i in zip(
                sel_columns, learning_model_functions[learning_model_type].feature_importances_)})
            print('Feature Importances:', importances_df.iloc[0].sort_values(ascending=False).to_dict())

        if perm_n_repeats > 0: _return_perm_imp(learning_model_functions[learning_model_type], features_df[sel_columns],
                                                features_df['is_active'], n_repeats=perm_n_repeats)
        # TODO: SHAP works only for trees currently
        if plot_SHAPLEY:   _plot_shap(learning_model_functions[learning_model_type], features_df[sel_columns])

        return learning_model_functions[learning_model_type], importances_df

    return _train_model

'''
You can get Linear SVC feature importances:
    https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
'''