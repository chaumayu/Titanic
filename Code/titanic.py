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

def main():
    train_path = "Data/titanic/train.csv"
    test_path = "Data/titanic/test.csv"

    train, test = Read(train_path, test_path)

    # Train.csv
    FillMissingValues(train)

if __name__ == '__main__':
    main()
