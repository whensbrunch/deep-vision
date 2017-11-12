from data import Data

def main():
    data = Data()
    print('X_train shape: ', data.X_train.shape)
    print('y_train length: ', len(data.y_train))
    print('X_test shape: ', data.X_test.shape)
    print('y_test length: ', len(data.y_test))

if __name__ == '__main__':
    main()