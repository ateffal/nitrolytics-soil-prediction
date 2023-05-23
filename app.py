import streamlit as st
import pandas as pd
import altair as alt
# from PIL import Image

import helpers as h
import compare_models as cm

# read data
water_recharge_df = pd.read_csv('data/water_recharge_data.csv')

# store columns
cols = list(water_recharge_df.columns)

st.write(
    """
    # Predicting Soil Characteristic
    ***
    """
)

st.write(
    """
    ### Sample data only - Will be linked to mvp-2 ( this data will be the one of the ROI selected )
    ***
    """
)

st.dataframe(water_recharge_df, 10000, 500)


var_target = st.selectbox(
    "Select target variable : ", cols, 2)


options = st.multiselect(
    "Select feature variables : ",
    cols, cols[3:5])


var_time = st.selectbox(
    "Select datetime variable - not used for now: ", cols)

st.sidebar.header('Choose a region - Not used for now')

continent = st.sidebar.selectbox(
    'Choose a continent', ['America', 'Europe', 'Africa', 'Asia', 'Austria'])


st.sidebar.header('Choose a city - Not used for now')

market = st.sidebar.selectbox(
    'Choose a city', ['city 1', 'city 2'])


choosen_model = st.sidebar.multiselect(
    'Compare models :', ['NNET', 'RNN', 'LSTM'], ['NNET', 'RNN'])


if not (var_target in options):
    vars_features = options + [var_target]
else:
    vars_features = options


selected_data = water_recharge_df[[var_time] + vars_features]

# st.write(
#     """
#     # Selected data :
#     ***
#     """
# )

st.write(" Target variable : " + var_target +
         ' -- features : ' + str(vars_features))


# st.dataframe(selected_data, 10000, 500)


if st.button('Predict'):
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

    predictions_2, rmses = cm.compare_models(selected_data,
                                             vars_features, var_target, choosen_model)

    st.line_chart(predictions_2)

    for m in choosen_model:
        st.text('Root Mean Squared Error of ' + m + ' : ' + str(rmses[m]))

    st.dataframe(predictions_2)
else:
    st.write(
        'Click Compare models to run the other models and compare - comparaison using test data')
