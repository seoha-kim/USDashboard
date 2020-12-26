from Bert.train import Train
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    BATCH_SIZE = 10
    EPOCHS = 3

    train = Train()
    X_train, X_test, Y_train, Y_test, model = train.preprocessing_reviews()
    with tf.device('/gpu:0'):
      opt = Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
      model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
      history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        
    model.save('../nlp_model.h5')