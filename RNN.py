from keras import Input
from keras.api.models import Sequential
from keras.api.layers import TextVectorization, Embedding, LSTM, Dense, Dropout
from keras.api.preprocessing import text_dataset_from_directory
from tensorflow._api.v2.strings import regex_replace


# Prepares data for training and testing
def prepare_data(dir):
    data = text_dataset_from_directory(dir)

    return data.map(
        lambda text, label: (regex_replace(text, '<br />', ' '), label)
    )

# Get prepared data
train_data = prepare_data("./movieSentData/train")
test_data = prepare_data("./movieSentData/test")

# Define model
model = Sequential()
model.add(Input(shape=(1,), dtype="string"))

# Defining TextVectorization layer
# Converts input text into vectors for embedding databse
max_tokens = 2000
max_len = 175
vectorization_layer = TextVectorization(
    # Max words processed at once
    max_tokens=max_tokens,
    # Output integer indices
    output_mode="int",
    # Consistent output shape
    output_sequence_length=max_len,
    )

train_texts = train_data.map(lambda text, label: text)
# fits layer to text data
vectorization_layer.adapt(train_texts)

model.add(vectorization_layer)

# Embedding layer (fixed length vectors)
model.add(Embedding(max_tokens + 1, 128))

# Recurrent Layer
model.add(LSTM(64))

model.add(Dropout(0.5))

# Dense layer for boolean output via sigmoid function
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy'],
)

# Train model
model.fit(train_data, epochs=10)

# Save weights
# model.save_weights('cnn.h5')

# # Load weights
# model.load_weights('cnn.h5')


