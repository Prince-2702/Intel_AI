from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dl_model import build_model

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    '../dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val = datagen.flow_from_directory(
    '../dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

model = build_model()
model.fit(train, validation_data=val, epochs=10)
model.save('../saved_model/visual_check.h5')
