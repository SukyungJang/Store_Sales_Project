# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
import utils

def feautreImportancePlot(model, X_train):
    # Get the best model from the search
    model = model.best_estimator_

    # Calculate the feature importances
    importances = model.feature_importances_ # 모델 선택 및 하이퍼파라미터 튜닝 기능을 사용하여 최적의 모델을 찾았을 때, 해당 모델 객체를 반환
    sorted_indices = np.argsort(importances)[::-1] # 값이 큰 순서대로 저장
    sorted_features = X_train.columns[sorted_indices]

    # Create a bar chart of feature importances
    fig = px.bar(x=sorted_features, y=importances[sorted_indices])
    fig.update_layout(
        title='Feature Importances',
        xaxis_title='Features',
        yaxis_title='Importance',
        plot_bgcolor='white',
        width=800,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

@st.cache_resource # Streamlit 캐시 데코레이터, 데이터 또는 모델 등의 리소스를 캐싱하는 데 사용
def run_model(data, max_depth, min_samples_leaf):
    # 특성과 타겟 분리

    y = data['sales']
    X = data.drop('sales', axis=1)

    # 훈련, 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write('Selected max_depth:', max_depth, '& min_samples_leat:', min_samples_leaf)

    random_search = {'max_depth': [i for i in range(max_depth[0], max_depth[1])],
                     'min_samples_leaf': [min_samples_leaf]}

    clf = RandomForestRegressor()
    model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 10,
                                   cv = 4, verbose= 1, random_state= 101, n_jobs = -1)
    model = model.fit(X_train, y_train)
    fig = feautreImportancePlot(model, X_train)

    return model, X_test, y_test, fig

def prediction(model, X_test, y_test):
    # 예측
    y_test_pred = model.predict(X_test)

    # 성능 평가
    test_mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    return y_test_pred, test_mae, r2

def prediction_plot(X_test, y_test, y_test_pred, test_mae, r2):
    # 그래프 그리기
    fig = go.Figure() # plotly express에서 제공하는 Figure 객체 생성 코드
    fig.add_trace(
        go.Scatter(x=X_test['transactions'], y=y_test, mode='markers', name='test', marker=dict(color='red'))
    )
    fig.add_trace(
        go.Scatter(x=X_test['transactions'], y=y_test_pred, mode='markers', name='prediction', marker=dict(color='green'))
    )

    fig.update_layout(
        title='Sales Prediction with RandomForestRegressor by Store Number',
        xaxis_title='Transactions',
        yaxis_title='Sales',
        annotations=[go.layout.Annotation(x=40, y=y_test.max(), text=f'Test MAE: {test_mae:.3f}<br>R2 Score: {r2:.3f}', showarrow=False)]
    )

    st.plotly_chart(fig)

def ml_app():
    # Hyperparameters
    max_depth = st.select_slider("Select max depth", options=[i for i in range(2, 30)], value=(5, 10), key='ml1')
    min_samples_leaf = st.slider("Minimum samples leaf", min_value=2, max_value=20, key='ml2')

    train, test, transactions, stores, oil, holidays = utils.load_data()

    df_data = train.merge(transactions, how='left', on=['date','store_nbr'])

    store_num = int(st.sidebar.number_input(label='store_nbr', step=1, min_value=1, max_value=df_data['store_nbr'].max()))

    data = pd.get_dummies(df_data.loc[df_data['store_nbr'] == store_num, ['family', 'transactions', 'sales']].dropna()) # one-hot 인코딩

    model, X_test, y_test, fig1 = run_model(data, max_depth, min_samples_leaf)
    y_test_pred, test_mae, r2 = prediction(model, X_test, y_test)

    prediction_plot(X_test, y_test, y_test_pred, test_mae, r2)

    st.markdown('<hr>', unsafe_allow_html=True)
    # Get the best model from the search
    st.plotly_chart(fig1)