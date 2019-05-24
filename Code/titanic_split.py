import regex as re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import lightgbm as lgb

from xgboost import XGBClassifier

def Read(train,test):
    train = pd.read_csv(train)
    test = pd.read_csv(test)

    print train.head()
    print train.info()

    # # Drop unique features values
    # train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    # test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    return train, test

def FillMissingValues(df):
    # Check for missing values
    print df.isnull().sum()

    # Embarked feature has some missing values and fill most occurred value ('S').
    print df['Embarked'].value_counts()
    embarked = df['Embarked'].value_counts().index[0]
    df['Embarked'] = df['Embarked'].fillna(embarked)

    # Age feature has some missing values and fill mean of age
    age_mean = int(df['Age'].mean())
    # print age_mean
    df['Age'] = df['Age'].fillna(age_mean)

    # Fare feature has some missing claues and fill mean of fare
    fare_mean = int(df['Fare'].mean())
    # print fare_mean
    df['Fare'] = df['Fare'].fillna(fare_mean)

    print df.isnull().sum()

    return df

def TrainFeatureEngineering(df):
    # Cabin
    df['HasCabin'] = df['Cabin'].apply(lambda x:0 if type(x) == float else 1)
    print df[['HasCabin', 'Survived']].groupby(['HasCabin'], as_index=False).mean()

    # PClass
    print df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

    # Sec
    print df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

    # SibSp + Parch (Sibling/Spouse + Parents/Childrens) = Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    print df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()

    # Alone or not
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    print df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

    # Embarked
    print df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()

    # Age
    # df['CatAge'] = pd.cut(df['Age'], 5)
    # # print df['CatAge'].head()
    # print df[['CatAge', 'Survived']].groupby(['CatAge'], as_index=False).mean()

    age_bins = [0,5,12,18,26,64,120]
    age_groups = ['Baby','Child','Teenager','Student','Adult','Senior']
    df['AgeGroup'] = pd.cut(df.Age, age_bins, labels = age_groups)
    # print df['AgeGroup'].head(10)
    print df[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean()

    # Fare
    # # print df['Fare'].min()
    # df['CatFare'] = pd.cut(df['Fare'], 4)
    # # print df['CatFare'].head()
    # print df[['CatFare', 'Survived']].groupby(['CatFare'], as_index=False).mean()

    fare_bins = [0,130,260,380,520]
    fare_groups = ['Low','Med','High','Very_High']
    df['FareGroup'] = pd.cut(df.Fare, fare_bins, labels = fare_groups)
    # print df['FareGroup'].head(10)
    print df[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean()


    # Name
    def getTitle(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
    	# If the title exists, extract and return it.
    	if title_search:
    		return title_search.group(1)
    	return ""

    df['Title'] = df['Name'].apply(getTitle)

    # print pd.crosstab(df['Title'], df['Sex'])

    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady',
                                        'Major', 'Rev', 'Sir'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    print df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    return df

def TestFeatureEngineering(df):
    # Cabin
    df['HasCabin'] = df['Cabin'].apply(lambda x:0 if type(x) == float else 1)

    # SibSp + Parch (Sibling/Spouse + Parents/Childrens) = Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone on ship
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Age
    age_bins = [0,5,12,18,26,64,120]
    age_groups = ['Baby','Child','Teenager','Student','Adult','Senior']
    df['AgeGroup'] = pd.cut(df.Age, age_bins, labels = age_groups)
    # print df['AgeGroup'].head(10)

    # Fare
    fare_bins = [0,130,260,380,520]
    fare_groups = ['Low','Med','High','Very_High']
    df['FareGroup'] = pd.cut(df.Fare, fare_bins, labels = fare_groups)
    # print df['FareGroup'].head(10)

    # Name
    def getTitle(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
    	# If the title exists, extract and return it.
    	if title_search:
    		return title_search.group(1)
    	return ""

    df['Title'] = df['Name'].apply(getTitle)

    # df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)

    # print pd.crosstab(df['Title'], df['Sex'])

    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady',
                                        'Major', 'Rev', 'Sir'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace('Dona', 'Mrs')

    return df

def DataCleaning(df):
    drop_columns = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare']
    df = df.drop(drop_columns, axis = 1)
    # print df.head()

    return df

def OneHotEncoding(df):
    df = pd.get_dummies(df, columns=['Pclass','Sex','Embarked','HasCabin','FamilySize','IsAlone',
                                        'AgeGroup','FareGroup','Title'])
    # print df.head()

    return df

def FeatureLabel(train):
    X_train = train.drop(['Survived'], axis=1)
    y_train = train['Survived']

    return X_train, y_train

# def DataScaling(X_train, X_test):
#     sc = StandardScaler()
#     # sc = RobustScaler()
#     X_train['Fare'] = sc.fit_transform(X_train[['Fare']])
#     # sc.fit(X_train)
#     # X_train_scaled = sc.transform(X_train)
#     # X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
#
#     X_test['Fare'] = sc.fit_transform(X_test[['Fare']])
#     # X_test_scaled = sc.transform(X_test)
#     # X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#
#     return X_train, X_test

def SupportVectorMachine(X_train, y_train, X_valid, y_valid):
    parameters = {  'kernel':('linear', 'rbf', 'poly'),
                    'C':[0.005, 0.01, 0.025, 0.05, 0.1, 1, 5],
                    'gamma':[0.001, 0.01, 0.1],
                    'degree':[3]}
    svc = SVC(random_state=0)
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    print clf.best_params_
    # {'kernel': 'linear', 'C': 1, 'gamma': 0.01}
    SVC_predict = clf.predict(X_valid)
    acc_score = metrics.accuracy_score(SVC_predict,y_valid)
    print 'Accuracy Score for SVM:-', acc_score
    # 0.8271186440677966

def RandomForest(X_train, y_train, X_valid, y_valid):
    parameters = {'max_depth'   : [4],#2, 4, 6
                    'max_features': [10],#8, 10, 12
                    "min_samples_split": [6]}#4, 6, 8
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0)
    clf = GridSearchCV(rf, parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    print clf.best_params_
    # {'max_features': 8, 'min_samples_split': 4, 'n_estimators': 300, 'max_depth': 4}
    # {'max_features': 8, 'min_samples_split': 2, 'max_depth': 4}
    # {'max_features': 10, 'min_samples_split': 6, 'max_depth': 4}

    RF_predict = clf.predict(X_valid)
    acc_score = metrics.accuracy_score(RF_predict,y_valid)
    print 'Accuracy Score for RF:-', acc_score
    # 0.8033898305084746
    # 500 - 0.8067796610169492
    # 1000 - 0.8169491525423729

    # print clf.feature_importances_

    return RF_predict

def XGBoost(X_train, y_train, X_valid, y_valid):
    parameters = {'gamma':[1], #0.5, 1, 1.5,
                'max_depth':[2], #2, 4, 6
                'min_child_weight':[1],# 1, 2, 5, 10
                'colsample_bytree': [0.4], # 0.6, 0.8, 1.0
                'subsample': [0.6], # 0.6, 0.8, 1
                'objective':['binary:logistic']}
    xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, nthread=4, random_state=0)
    clf = GridSearchCV(xgb, parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    print clf.best_params_
    # {'colsample_bytree': 0.6, 'min_child_weight': 1, 'subsample': 0.8,
    #             'objective': 'binary:logistic', 'max_depth': 2, 'gamma': 1.5}
    # {'colsample_bytree': 0.4, 'min_child_weight': 1, 'subsample': 0.6,
    #             'objective': 'binary:logistic', 'max_depth': 2, 'gamma': 1}
    # {'colsample_bytree': 0.2, 'min_child_weight': 1, 'subsample': 0.6,
    #             'objective': 'binary:logistic', 'max_depth': 3, 'gamma': 0.5}

    XGB_predict = clf.predict(X_valid)
    acc_score = metrics.accuracy_score(XGB_predict,y_valid)
    print 'Accuracy Score for XGB:-', acc_score
    # 0.823728813559322
    # 100 - 0.8271186440677966 1000 - 0.8305084745762712
    # 0.8203389830508474

    # print clf.feature_importances_
    # feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    # feat_importances.nlargest(10).plot(kind='barh')
    # plt.show()

    return XGB_predict

def DecisionTree(X_train, y_train, X_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    DT_predict = clf.predict(X_test)

    # print clf.feature_importances_

    return DT_predict

def GradientBoosting(X_train, y_train, X_valid, y_valid):
    parameters = {'loss' : ["deviance","exponential"],
                    'max_depth':  [4],# 3, 4, 5
                    'max_features': [3], # 2, 3, 4
                    'min_samples_leaf': [5]} # 4, 5, 6
    gb = GradientBoostingClassifier(n_estimators = 300, learning_rate=0.01, random_state=0)
    clf = GridSearchCV(gb, parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    print clf.best_params_
    # {'max_features': 3, 'loss': 'deviance', 'max_depth': 4, 'min_samples_leaf': 5}

    GB_predict = clf.predict(X_valid)
    acc_score = metrics.accuracy_score(GB_predict,y_valid)
    print 'Accuracy Score for GB:-', acc_score
    # 0.8372881355932204

    # print clf.feature_importances_

    return GB_predict

def main():
    train_path = "Data/titanic/train.csv"
    test_path = "Data/titanic/test.csv"

    train, test = Read(train_path, test_path)

    # Train.csv
    train_fillna = FillMissingValues(train)
    train_feature = TrainFeatureEngineering(train_fillna)
    train_clean = DataCleaning(train_feature)
    train_ohe = OneHotEncoding(train_clean)
    X_train, y_train = FeatureLabel(train_ohe)
    # print X_train.info()

    test_fillna = FillMissingValues(test)
    test_feature = TestFeatureEngineering(test_fillna)
    test_clean = DataCleaning(test_feature)
    X_test = OneHotEncoding(test_clean)
    # print X_test.info()

    # X_train, X_test = DataScaling(X_train, X_test)
    # X_train.to_csv('Results/train-processed.csv', index=False)
    # X_test.to_csv('Results/test-processed.csv', index=False)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                            test_size=0.33, random_state=0)

    # SVC_predict = SupportVectorMachine(X_train, y_train, X_valid, y_valid)

    # RF_predict = RandomForest(X_train, y_train, X_valid, y_valid)

    # XGB_predict = XGBoost(X_train, y_train, X_valid, y_valid)

    # DT_predict = DecisionTree(X_train, y_train, X_test)

    GB_predict = GradientBoosting(X_train, y_train, X_valid, y_valid)

    # submission = pd.DataFrame({"PassengerId": test_feature['PassengerId'], "Survived": XGB_predict})
    # submission.to_csv('Results/submission_xgb3.csv', index=False)

if __name__ == '__main__':
    main()
