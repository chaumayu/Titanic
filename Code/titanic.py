import pandas as pd

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
    print df.isnull().any()

    # Embarked feature has some missing values and fill most occurred value ('S').
    print df['Embarked'].value_counts()
    embarked = df['Embarked'].value_counts().index[0]
    df['Embarked'] = df['Embarked'].fillna(embarked)

    # Age feature has some missing values and fill mean of age
    age_mean = int(df['Age'].mean())
    # print age_mean
    df['Age'] = df['Age'].fillna(age_mean)

    print df.isnull().any()

def FeatureEngineering(df):
    # PClass with Survived
    print df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

    # Sec with Survived
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
    Age_bins = [0,5,12,18,26,60,120]
    Age_groups = ['Baby','Child','Teenager','Student','Adult','Senior']
    df['AgeGroup'] = pd.cut(df.Age, Age_bins, labels = Age_groups)
    print df['AgeGroup'].head(10)

    # Fare
    # Name

def main():
    train_path = "Data/titanic/train.csv"
    test_path = "Data/titanic/test.csv"

    train, test = Read(train_path, test_path)

    # Train.csv
    FillMissingValues(train)
    FeatureEngineering(train)

if __name__ == '__main__':
    main()
