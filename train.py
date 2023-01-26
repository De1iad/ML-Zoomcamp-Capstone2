from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_ds = train_gen.flow_from_directory(
    './split/train', seed=42, target_size=(299, 299), batch_size=32, class_mode='categorical')

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory('./split/val', seed=42, target_size=(299, 299), batch_size=32, shuffle=False)

checkpoint = keras.callbacks.ModelCheckpoint('xception_{epoch:02d}_{val_accuracy:.3f}.h5',
                                            save_best_only=True,
                                            monitor='val_accuracy',
                                            mode='max')

learning_rate = 0.001
size_inner = 1000
droprate = 0.2
base_model = Xception(weights='imagenet', input_shape=(299, 299, 3), include_top=False)
base_model.trainable=False
inputs = keras.Input(shape=(299, 299, 3))
base = base_model(inputs, training=False)
vectors = keras.layers.GlobalAveragePooling2D()(base)
inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
drop = keras.layers.Dropout(droprate)(inner)
outputs = keras.layers.Dense(102)(drop)
model = keras.Model(inputs, outputs)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[checkpoint])