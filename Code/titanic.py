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

def main():
    train_path = "Data/titanic/train.csv"
    test_path = "Data/titanic/test.csv"

    train, test = Read(train_path, test_path)

if __name__ == '__main__':
    main()
