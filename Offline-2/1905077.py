import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# datapaths
adult_train = 'adult/adult.data'
adult_test = 'adult/adult.test'
churn ='churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
credit_card = 'creditcard/creditcard.csv'

# preprocess function, returns X_train, X_test, y_train, y_test for each choice
def preprocess(choice):
    if choice == 'adult':
        df_train = pd.read_csv(adult_train, header=None)
        df_test = pd.read_csv(adult_test, header=None, skiprows=1) # skip the first row

        # ? to NaN
        df_train = df_train.replace(' ?', np.nan)
        df_test = df_test.replace(' ?', np.nan)

        # drop NaN
        # df = df.dropna()
        # df_test = df_test.dropna()

        #fill NaN with mode for categorical and mean for numerical
        for df in [df_train, df_test]:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].mean())
            
        # target column is like '>50K' and '<=50K', convert to 1 and 0
        df_train[14] = df_train[14].apply(lambda x: 1 if x == ' >50K' else 0)
        df_test[14] = df_test[14].apply(lambda x: 1 if x == ' >50K.' else 0)

        # split X and y
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

        # label encoding for categorical columns with binary values
        for df in [X_train, X_test]:
            for col in df.columns:
                if df[col].dtype == 'object' and len(df[col].unique()) > 2:
                    # one hot encoding
                    df = pd.get_dummies(df, columns=[col], drop_first=True)
                else:
                    # label encoding
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
            

        for df in [X_train, X_test]:
            for col in df.columns:
                if len(df[col].unique()) > 2 and df[col].dtype != 'object':
                    scaler = StandardScaler()  
                    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))  

        return X_train, X_test, y_train, y_test

    elif choice == 'credit_card':
        df = pd.read_csv(credit_card)
        #drop time column
        df = df.drop('Time', axis=1)
        #randomly selected 20000 negative samples + all positive samples
        df = pd.concat([df[df['Class'] == 0].sample(20000, random_state=42), df[df['Class'] == 1]])
        # shuffle
        df = df.sample(frac=1, random_state=42)

        # split X and y
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # label encoding for categorical columns with binary values
        for col in X.columns:
            if len(X[col].unique()) > 2:
                # one hot encoding
                X = pd.get_dummies(X, columns=[col], drop_first=True)
            else:
                # label encoding
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        for col in X.columns:
            if len(X[col].unique()) > 2:
                scaler = StandardScaler()  
                X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


# logistic regression class
class LogisticRegression:
    def __init__(self,learning_rate=0.01, max_epoch=1000):
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.max_epoch):
            # forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # loss
            loss = self.loss(y, y_pred)

            # gradient
            dw = np.dot(X.T, (y_pred - y)) / X.shape[0]
            db = np.mean(y_pred - y)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y

    def loss(self, y, y_pred):
        y_pred= np.where(y_pred == 0, 1e-15, y_pred)
        y_pred= np.where(y_pred == 1, 1-1e-15, y_pred)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # return Accuracy Sensitivity Specificity Precision F1-score AUCROC AUPRC
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        TP = np.sum((y == 1) & (y_pred == 1))
        TN = np.sum((y == 0) & (y_pred == 0))
        FP = np.sum((y == 0) & (y_pred == 1))
        FN = np.sum((y == 1) & (y_pred == 0))

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)
        
        return accuracy, sensitivity, specificity, precision, f1_score

# main function
def main(choice):
    choice = 'adult'
    X_train, X_test, y_train, y_test = preprocess(choice)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy, sensitivity, specificity, precision, f1_score = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'Precision: {precision}')
    print(f'F1-score: {f1_score}')

if __name__ == '__main__':
    main('adult')