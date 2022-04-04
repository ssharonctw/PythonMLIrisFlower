#the app uses randomForest algorithm to predict the irs flower based on 4 input paramters
#the model reaches accuracy at 92%+ when the #ofTrees were set as 100, min_sample_split = 2

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

#a custom function used to accept all of the four input parameter from the side bar in a dictionary
#then we'll use a panda DataFrame to store the input parameters in the dictionary

#using streamlit.sidebar to place content in the sidebar
#use sidebar.slider to get the user's input of the four parameters
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()

X = iris.data #the datasets of the four attributes
Y = iris.target #the result corresponded by the four attributes above

#test-size = 25% of data is test datasets
#stratify = y tells train_test_split to contain examples of each class in the same proportion as the original dataset
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=77, stratify = iris.target)

#n_estimators = # of trees created
#min_sample_split = #of samples required to split an internal node
clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=77)
clf.fit(x_train, y_train)



# Make predictions for the test set
y_pred_test = clf.predict(x_test)

# View accuracy score
acc=accuracy_score(y_test, y_pred_test)

st.subheader("Current Model's Accuracy:")
st.write(acc)
#
#result of prediction model
prediction = clf.predict(df)
#
#the probabilty of each all possible results (the above result is the possbile result with highest probability)
prediction_proba = clf.predict_proba(df)
#
#
#
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)
#
st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)
#
st.subheader('Prediction Probability')
st.write(prediction_proba)
