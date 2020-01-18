submission = pd.read_csv("sample_submission.csv")
#the model_file needs to be trained with kaggle data
model.load_weights('model_file')

# Create a generator from the submission dataframe to leverage model.predict_generator to
# make the predictions

test_datagen = ImageDataGenerator(rescale=1./255)
# the config here needs to be the same as train_generator and i think it is better to turn the shuffle off.
test_generator = test_datagen.flow_from_dataframe(#if we want to use stanford data we need to create another submission file 
                                                dataframe=submission,
                                                directory=TEST_DIR,
                                                x_col='id',
                                                class_mode=None,
                                                has_ext=False,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                seed=SEED,
                                                target_size=(INPUT_SIZE, INPUT_SIZE)
                                               )

predictions = model.predict_generator(test_generator, verbose=1)


# Substitute the dummy predictions in submmission by the model predictions,
submission.loc[:,1:] = predictions
submission.head()