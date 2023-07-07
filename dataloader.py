from keras.preprocessing.image import ImageDataGenerator

def load_data():
    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                      shear_range = 0.2, 
                                      zoom_range = 0.2, 
                                      horizontal_flip = True,
                                      rotation_range=20,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    spiral_train_generator = train_datagen.flow_from_directory('C:/huh/pd spiral/archive/drawings/spiral/training',
                                                       target_size = (128,128),
                                                       batch_size = 32,
                                                       class_mode = 'binary')

    spiral_test_generator = test_datagen.flow_from_directory('C:/huh/pd spiral/archive/drawings/spiral/testing',
                                                       target_size = (128,128),
                                                       batch_size = 32,
                                                       class_mode = 'binary')

    return spiral_train_generator, spiral_test_generator
