import streamlit as st
import pandas as pd
import altair as alt
# from PIL import Image

import helpers as h
import compare_models as cm

# read data
# water_recharge_df = pd.read_csv('data/water_recharge_data.csv')


st.write(
    """
    # Predicting Soil Characteristic
    ***
    """
)

# the user can load a csv file
uploaded_file = st.file_uploader("Choose a csv file")

if uploaded_file is not None:
    water_recharge_df = pd.read_csv(uploaded_file)
    st.dataframe(water_recharge_df, 10000, 500)

    # store columns
    cols = list(water_recharge_df.columns)

    # select the target variable
    var_target = st.selectbox(
        "Select target variable : ", cols, 2)

    # select the predictors
    options = st.multiselect("Select feature variables : ", cols, cols[3:5])

    # select time column
    var_time = st.selectbox(
        "Select datetime variable - not used for now: ", cols)

    if not (var_target in options):
        vars_features = options + [var_target]
    else:
        vars_features = options

    # selected data
    selected_data = water_recharge_df[[var_time] + vars_features]


st.sidebar.header('Choose a region - Not used for now')

continent = st.sidebar.selectbox(
    'Choose a continent', ['America', 'Europe', 'Africa', 'Asia', 'Austria'])


st.sidebar.header('Choose a city - Not used for now')

market = st.sidebar.selectbox(
    'Choose a city', ['city 1', 'city 2'])


choosen_model = st.sidebar.multiselect(
    'Compare models :', ['NNET', 'RNN', 'LSTM'], ['NNET', 'RNN'])


if uploaded_file is not None:
    if st.button('Predict'):
        st.write(" Target variable : " + var_target +
                 ' -- features : ' + str(vars_features))

        selected_data, predictions, RMSE = h.prediction_nnet_dense_layers(selected_data,
                                                                          features=vars_features,
                                                                          target=var_target,
                                                                          lags=6,
                                                                          scale_data=True)
        st.write(
            """
        # Predictions  
        Sequential Neural Network with three hidden layers : 32 nodes, 32 nodes, 32 nodes \n
        6 lags of the target variable
        """ + var_target
        )

        st.line_chart(
            predictions[[predictions.columns[0], predictions.columns[1]]])

        st.dataframe(predictions)

        st.write('Root Mean Squared Error : ' + str(RMSE))

    else:
        st.write('Click Predict to run model')

    if st.button('Compare models'):

        st.write(
            """
            # Compare models by splitting data into train, validation and test :
            ***
            """
        )

        pred_train, pred_val, pred_test, models = cm.compare_models(selected_data,
                                                                    vars_features, var_target, choosen_model)

        st.line_chart(pred_train[0])

        st.text('Train data :')
        for m in choosen_model:
            st.text('Root Mean Squared Error of ' +
                    m + ' : ' + str(pred_train[1][m]))

        st.text('Validation data :')
        for m in choosen_model:
            st.text('Root Mean Squared Error of ' +
                    m + ' : ' + str(pred_val[1][m]))

        st.text('Test data :')
        for m in choosen_model:
            st.text('Root Mean Squared Error of ' +
                    m + ' : ' + str(pred_test[1][m]))

        st.text('Predictions for train data :')
        st.dataframe(pred_train[0])

        st.text('Predictions for validation data :')
        st.dataframe(pred_val[0])

        st.text('Predictions for test data :')
        st.dataframe(pred_test[0])

    else:
        st.write(
            'Click Compare models to run the other models and compare - comparaison using test data')
