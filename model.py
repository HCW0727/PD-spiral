from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_model():
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Flatten())

    classifier.add(Dense(512, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.5))

    classifier.add(Dense(128, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.5))

    classifier.add(Dense(1, activation='sigmoid'))

    return classifier
