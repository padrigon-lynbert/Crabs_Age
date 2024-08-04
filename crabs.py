import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_validate

from connections import get_df

df = get_df()
df['Sex'] = df['Sex'].map({'F':2, 'M':1, 'I':0})

X = df.drop('Age', axis=1)
y = df['Age']


ensemble = VotingRegressor([
    ('catboost', CatBoostRegressor(random_state=42, silent=True)),
    ('svr', SVR()),
    ('knn', KNeighborsRegressor())
])

pipeline_ensemble = Pipeline([
    ('scaler', StandardScaler()),
    ('ensemble', ensemble)
])

classification_result_ensemble = cross_validate(pipeline_ensemble, X, y, scoring=['neg_mean_squared_error', 'r2'])


mse = -classification_result_ensemble['test_neg_mean_squared_error']
rmse = np.sqrt(mse)
r2 = classification_result_ensemble['test_r2']


print(mse.mean(), rmse.mean(), r2.mean()) 
    # 4.62482579284704 2.1491025590805157 0.5529300209853195