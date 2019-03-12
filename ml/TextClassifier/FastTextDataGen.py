# coding:utf-8
import pandas as pd
import random



def read_dataset(data_path, option = 1):
    # Data Preparation
    # ==================================================

    # Load data

    if option == 0:
        text_col = 'rough_token'
    elif option == 1:
        text_col = 'rigour_token'
    else:
        raise ('option par error! (must be 0 or 1)')

    print("Loading data...")
    df = pd.read_csv(data_path)

    df.dropna(how='any', inplace=True)

    train_data = df[df['train_val_test'] != 3]
    test_data = df[df['train_val_test'] == 3]

    # Train set
    x_train = list(map(lambda x: x.replace('。', ' '), train_data[text_col].tolist()))
    y_train = train_data['cls'].tolist()

    train_set = []
    for x, y in zip(x_train, y_train):
        train_set.append("__lable__" + str(int(y)) + " , " + x)

    # Test set
    x_test = list(map(lambda x: x.replace('。', ' '), test_data[text_col].tolist()))
    y_test = test_data['cls'].tolist()

    test_set = []
    for x, y in zip(x_test, y_test):
        test_set.append("__lable__" + str(int(y)) + " , " + x)

    return train_set, test_set


def writeData(sentences, fileName):
    random.shuffle(sentences)
    print("writing data to fasttext format...")
    with open(fileName, 'w', encoding='utf-8') as fp:
        for sentence in sentences:
            fp.write(sentence+"\n")
    print("done!")



if __name__=="__main__":
    data_file = r'../data/trainSet/train_info_5w.csv'

    train_set_file = r'../data/trainSet/fastText/trainData.txt'
    test_set_file = r'../data/trainSet/fastText/testData.txt'

    train_set, test_set = read_dataset(data_file)
    writeData(train_set, train_set_file)
    writeData(test_set, test_set_file)

