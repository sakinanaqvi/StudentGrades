import pandas as pd 
import joblib
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def load_data():
# loading the dataset
    data = pd.read_csv('student-mat.csv', sep=';')
    print(data.head())
    print(data.columns)
    return data

# will use studytime, failures, G1, G2, famrel, freetime, goout

def convert_data(data):
    #convert categorical data to numerical
    data_encoded = pd.get_dummies(data, drop_first=True)
    print(data_encoded.head())

    #splitting the data into features and target - target is the final grade
    X = data_encoded.drop(columns=['G3'])

    y = data_encoded['G3']
    return X, y

# splitting into training and testing
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# training 
def train(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# evaluate the model
def evaluate(model, X_test, y_test):
    y_prediction = model.predict(X_test)

    #calculate MAE and r^2
    r2 = r2_score(y_test, y_prediction)
    mae = mean_absolute_error(y_test, y_prediction)

    print(f'Mean Absolute Error: {mae}')
    print(f'R-squared: {r2}')

    # scatterplot to show the accuract 
   # plt.scatter(y_test, y_prediction)
   # plt.xlabel('Actual Grades')
   # plt.ylabel('Predicted Grades')
   # plt.title('Actual vs Predicted Grades')
   # plt.show()

def save(model): 
    joblib.dump(model, 'student_grade_predictor.pkl')

# make predictions based on new data
def predict_data(model, new_data, train_columns):
    new_data_encoded = pd.get_dummies(new_data, drop_first=True)
    new_data_encoded = new_data_encoded.reindex(columns=train_columns, fill_value=0)

    predicted_grade = model.predict(new_data_encoded)
    print(f'Predicted Final Grade: {predicted_grade[0]}')

def main():
    data = load_data()
    X, y = convert_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)
    save(model)

    new_data = pd.DataFramenew_data = pd.DataFrame({
        'studytime': [5],         
        'failures': [0],           
        'famrel': [5],             
        'freetime': [3],           
        'goout': [4],             
        'sex_M': [1],              
        'address_U': [0],          
        'G1': [12],                
        'G2': [14],               
    })

    predict_data(model, new_data, X.columns)

if __name__ == '__main__':
    main()
