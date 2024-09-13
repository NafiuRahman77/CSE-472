import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

adult_train = 'adult/adult.data'
adult_test = 'adult/adult.test'
churn ='churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
credit_card = 'creditcard/creditcard.csv'

def preprocess(choice):
    if choice == 'adult':
        df_train = pd.read_csv(adult_train, header=None)
        df_test = pd.read_csv(adult_test, header=None, skiprows=1) # skip the first row

        df_train = df_train.replace(' ?', np.nan)
        df_test = df_test.replace(' ?', np.nan)

        # drop NaN
        # df = df.dropna()
        # df_test = df_test.dropna()

        for col in df_train.columns:
            if df_train[col].dtype == 'object':
                df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
            else:
                df_train[col] = df_train[col].fillna(df_train[col].mean())

        for col in df_test.columns:
            if df_test[col].dtype == 'object':
                df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
            else:
                df_test[col] = df_test[col].fillna(df_test[col].mean())

        df_train[14] = df_train[14].apply(lambda x: 1 if x == ' >50K' else 0)
        df_test[14] = df_test[14].apply(lambda x: 1 if x == ' >50K.' else 0)
        
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

            
        for col in X_train.columns:
            if X_train[col].dtype == 'object' and len(X_train[col].unique()) == 2:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
            
            elif X_train[col].dtype == 'object':
                X_train = pd.get_dummies(X_train, columns=[col], drop_first=True)

        for col in X_test.columns:
            if X_test[col].dtype == 'object' and len(X_test[col].unique()) == 2:
                le = LabelEncoder()
                X_test[col] = le.fit_transform(X_test[col])
            
            elif X_test[col].dtype == 'object':
                X_test = pd.get_dummies(X_test, columns=[col], drop_first=True)

        #add missing columns to test set
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[X_train.columns]

        for col in X_train.columns.values:
            scalar = StandardScaler()
            X_train[col] = scalar.fit_transform(X_train[col].values.reshape(-1, 1))
            X_test[col] = scalar.transform(X_test[col].values.reshape(-1, 1))

        return X_train, X_test, y_train, y_test

    elif choice == 'credit_card':
        df = pd.read_csv(credit_card)
        df = df.drop('Time', axis=1)
        #randomly selected 20000 negative samples + all positive samples
        df = pd.concat([df[df['Class'] == 0].sample(20000, random_state=42), df[df['Class'] == 1]])
        # shuffle
        df = df.sample(frac=1, random_state=42)

        # split X and y
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        for col in X.columns:
            if len(X[col].unique()) > 2 and X[col].dtype == 'object':

                X = pd.get_dummies(X, columns=[col], drop_first=True)
            elif  X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        for col in X.columns:
            if len(X[col].unique()) > 2:
                scaler = StandardScaler()  
                X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    elif choice == 'churn':
        df = pd.read_csv(churn)
        df = df.drop('customerID', axis=1)

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])  # Fill NaN for categorical columns
            else:
                df[col] = df[col].fillna(df[col].mean())  # Fill NaN for numeric columns

        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        for col in X.columns:
            if len(X[col].unique()) > 2 and X[col].dtype == 'object':
                X = pd.get_dummies(X, columns=[col], drop_first=True)
            elif X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        for col in X.columns:
            if len(X[col].unique()) > 2:
                scaler = StandardScaler()  
                X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

        # change boolean columns to binary
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].apply(lambda x: 1 if x == True else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

class LogisticRegression:
    def __init__(self,learning_rate=0.01, max_epoch=1000):
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.array(z)
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
        return y_pred

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
    main('churn')