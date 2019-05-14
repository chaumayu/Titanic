import regex as re
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

    return df

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
    # age_bins = [0,5,12,18,26,60,120]
    # age_groups = ['Baby','Child','Teenager','Student','Adult','Senior']
    # df['AgeGroup'] = pd.cut(df.Age, age_bins, labels = age_groups)
    # # print df['AgeGroup'].head(10)
    # print df[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean()

    df['CatAge'] = pd.cut(df['Age'], 5)
    # print df['CatAge'].head()
    print df[['CatAge', 'Survived']].groupby(['CatAge'], as_index=False).mean()

    # Fare
    df['CatFare'] = pd.cut(df['Fare'], 4)
    # print df['CatFare'].head()
    print df[['CatFare', 'Survived']].groupby(['CatFare'], as_index=False).mean()

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

def main():
    train_path = "Data/titanic/train.csv"
    test_path = "Data/titanic/test.csv"

    train, test = Read(train_path, test_path)

    # Train.csv
    train = FillMissingValues(train)
    train = FeatureEngineering(train)

if __name__ == '__main__':
    main()
