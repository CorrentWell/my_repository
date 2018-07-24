from keras.models.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers

model = VGG16(include_top=False, weights='imagenet')

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(3, activation='softmax'))

# VGG16とFCを接続
model = model(input=model.input, output=top_model(model.output))

# 最後のconv層の直前までの層をfreeze
for layer in model.layers[:15]:
    layer.trainable = False

# Fine-tuningのときはSGDの方がよい
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])