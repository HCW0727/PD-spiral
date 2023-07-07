import model
import dataloader
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

classifier = model.create_model()

spiral_train_generator, spiral_test_generator = dataloader.load_data()

early_stopping = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks_list = [early_stopping,reduce_learningrate]

classifier.compile(loss='binary_crossentropy',
              optimizer = Adam(lr=0.0001),
              metrics=['accuracy'])

history = classifier.fit_generator(
        spiral_train_generator,
        steps_per_epoch=spiral_train_generator.n//spiral_train_generator.batch_size,
        epochs=400,
        validation_data=spiral_test_generator,
        validation_steps=spiral_test_generator.n//spiral_test_generator.batch_size,
        callbacks=callbacks_list)
