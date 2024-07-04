import pickle
import numpy as np
import os
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation

def generate_music(output_folder='generated music/'):
    """ Generate MIDI notes using a trained neural network """
    # Load the notes used to train the model
    with open('data/notes', 'rb') as file:
        notes_data = pickle.load(file)

    # Get all unique pitch names
    pitch_names = sorted(set(item for item in notes_data))
    # Get the number of unique pitch names
    n_vocab = len(set(notes_data))

    network_input, normalized_input = prepare_sequences(notes_data, pitch_names, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitch_names, n_vocab)
    create_midi(prediction_output, output_folder)

def prepare_sequences(notes_data, pitch_names, n_vocab):
    """ Prepare sequences used by the Neural Network """
    # Map between notes and integers
    note_to_int = {note: number for number, note in enumerate(pitch_names)}

    sequence_length = 100
    network_input = []
    output = []
    for i in range(len(notes_data) - sequence_length):
        in_sequence = notes_data[i:i + sequence_length]
        out_sequence = notes_data[i + sequence_length]
        network_input.append([note_to_int[char] for char in in_sequence])
        output.append(note_to_int[out_sequence])

    n_patterns = len(network_input)

    # Reshape input for LSTM
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input

def create_network(normalized_input, n_vocab):
    """ Create the LSTM Neural Network """
    model = Sequential([
        LSTM(512, input_shape=(normalized_input.shape[1], normalized_input.shape[2]), recurrent_dropout=0.3, return_sequences=True),
        LSTM(512, return_sequences=True, recurrent_dropout=0.3),
        LSTM(512),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load weights into the model
    model.load_weights('weights.hdf5')

    return model

def generate_notes(model, network_input, pitch_names, n_vocab):
    """ Generate notes from the Neural Network """
    start_idx = np.random.randint(0, len(network_input) - 1)

    int_to_note = {number: note for number, note in enumerate(pitch_names)}

    pattern = network_input[start_idx]
    prediction_output = []

    # Generate 500 notes
    for _ in range(500):
        input_pattern = np.reshape(pattern, (1, len(pattern), 1))
        input_pattern = input_pattern / float(n_vocab)

        prediction = model.predict(input_pattern, verbose=0)

        idx = np.argmax(prediction)
        result = int_to_note[idx]
        prediction_output.append(result)

        pattern.append(idx)
        pattern = pattern[1:]

    return prediction_output

def create_midi(prediction_output, output_folder):
    """ Convert prediction to MIDI notes and create a MIDI file """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(current_note)) for current_note in notes_in_chord]
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.storedInstrument = instrument.Guitar()
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Guitar()
            output_notes.append(new_note)

        offset += 0.5

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    midi_stream = stream.Stream(output_notes)

    # Save MIDI file
    midi_file_path = os.path.join(output_folder, 'generated_music.mid')
    midi_stream.write('midi', fp=midi_file_path)

if __name__ == '__main__':
    generate_music()
