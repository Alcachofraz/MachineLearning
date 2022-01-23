from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import pygame.midi as midi
import pygame
from pygame.locals import *

"""
---- CONTEXTUALIZAÇÃO ----

Na música, os acordes fundamentais são compostos por 3 notas.
A tónica, a terceira e a quinta.

Existem acordes maiores e menores.

Em qualquer acorde, a quinta dista 7 meio-tons da tónica.

Nos acordes maiores, a terceira dista 4 meio-tons da tónica.
Nos acordes menores, a terceira dista 3 meio-tons da tónica.

Uma oitava conta com 12 notas diferentes.
Existem 24 acordes possíveis numa oitava (de Dó a Si).


---- CAMADA DE ENTRADA ----

Uma oitava será representada pela seguinte lista:
[*, *, *, *, *, *, *, *, *, *, *, *]
Os dados de entrada terão este formato, em que as posições
ativadas (a 1) representarão notas a soar, e as posições
desativadas (a 0) representarão notas em silêncio.


---- CAMADA DE SAÍDA ----

Dado que haverão 24 acordes possíveis, a camada de saída
contará com 24 neurónios, um para cada acorde. Os primeiros
12 serão maiores (de Dó a Si) e os segundos 12 serão menores
(também de Dó a Si).


---- EXEMPLO ----

Acorde Dó Maior (Dó, Mi, Sol)
Entrada:
[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
Saída:
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Acorde Sol Menor (Sol, Lá Sustenido, Ré)
Entrada:
[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
Saída:
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

"""

USING_MIDI = None

HIDDEN_LAYERS = (1024, 512)
ACTIVATION = 'relu'
SOLVER = 'adam'
VERBOSE = False

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

NUMBER_OF_CHORDS = 24
NUMBER_OF_MINOR_CHORDS = 12
NUMBER_OF_MAJOR_CHORDS = 12
NOTES_PER_CHORD = 3
NOTES_PER_OCTAVE = 12

MAJOR_CHORD_PATTERN = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
MINOR_CHORD_PATTERN = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

OUTPUT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

FORCED_CHORDS_NUM = 100  # Per chord
RANDOM_CHORDS_NUM = 2400
TOTAL_DATA_LENGTH = (FORCED_CHORDS_NUM * NUMBER_OF_CHORDS) + RANDOM_CHORDS_NUM

CHORD_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
               'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm', ]

NOTES = ['C.ogg', 'C#.ogg', 'D.ogg', 'D#.ogg',
         'E.ogg', 'F.ogg', 'F#.ogg', 'G.ogg', 'G#.ogg', 'A.ogg', 'A#.ogg', 'B.ogg', ]


def rotate(octave, n):
    """
    Takes an array [list] (octave) and rotates its positions [n] times.
    """
    return [octave[(i - n) % len(octave)]
            for i, x in enumerate(octave)]


def is_chord(octave):
    """
    Takes an array [list] (octave) and returns a 24 length list indicating
    which chord is being played, if any.
    """
    ret = OUTPUT.copy()
    if np.count_nonzero(octave) != NOTES_PER_CHORD:
        return ret
    for i in range(NOTES_PER_OCTAVE):
        temp = rotate(octave, -i)
        if temp[0] == 1 and temp[7] == 1:
            if temp[3] == 1:
                ret[i + NOTES_PER_OCTAVE] = 1
                return ret
            elif temp[4] == 1:
                ret[i] = 1
                return ret
    return ret


def random_chord():
    return [[rnd.randint(0, 1) for i in range(NOTES_PER_OCTAVE)]]


def print_devices():
    for n in range(midi.get_count()):
        print(n, midi.get_device_info(n))


def to_chord(octave):
    if octave.count(1) == 1:
        return CHORD_NAMES[octave.index(1)]
    else:
        return 'UNKNOWN'


def event_to_chord(event):
    if event.key == pygame.K_q:
        return 0
    elif event.key == pygame.K_2:
        return 1
    elif event.key == pygame.K_w:
        return 2
    elif event.key == pygame.K_3:
        return 3
    elif event.key == pygame.K_e:
        return 4
    elif event.key == pygame.K_r:
        return 5
    elif event.key == pygame.K_5:
        return 6
    elif event.key == pygame.K_t:
        return 7
    elif event.key == pygame.K_6:
        return 8
    elif event.key == pygame.K_y:
        return 9
    elif event.key == pygame.K_7:
        return 10
    elif event.key == pygame.K_u:
        return 11
    else:
        return -1


"""
-----------
Train Model
-----------
"""
print('Learning...')

# Initialise X (training data) empty:
X = np.array([], dtype=int)

# Initialise Y (target data) empty:
Y = np.array([], dtype=int)

# Append to X all of the [NUMBER_OF_MAJOR_CHORDS] major chords, [FORCED_CHORDS_NUM] times each:
for i in range(NUMBER_OF_MAJOR_CHORDS):
    X = np.append(X, rotate(MAJOR_CHORD_PATTERN, i) * FORCED_CHORDS_NUM)

# Append to X all of the [NUMBER_OF_MINOR_CHORDS] minor chords, [FORCED_CHORDS_NUM] times each:
for i in range(NUMBER_OF_MINOR_CHORDS):
    X = np.append(X, rotate(MINOR_CHORD_PATTERN, i) * FORCED_CHORDS_NUM)

# Append to X [RANDOM_CHORDS_NUM] random chords:
for i in range(RANDOM_CHORDS_NUM):
    X = np.append(X, random_chord())

X = np.reshape(X, (TOTAL_DATA_LENGTH, NOTES_PER_OCTAVE))

# Shuffle X:
np.random.shuffle(X)

# Append target data:
for i in range(TOTAL_DATA_LENGTH):
    Y = np.append(Y, is_chord(X[i]))

Y = np.reshape(Y, (TOTAL_DATA_LENGTH, NUMBER_OF_CHORDS))

# Train model:
regr = MLPRegressor(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation=ACTIVATION,
    solver=SOLVER,
    verbose=VERBOSE,
)
model = regr.fit(X, Y)

print('Model is ready!')


"""
--------
Ask Midi
--------
"""
while USING_MIDI == None:
    msg = input('Do you want to use MIDI input? [y/n]')
    if msg == 'y':
        USING_MIDI = True
    elif msg == 'n':
        USING_MIDI = False
    else:
        print('Invalid input.')

if USING_MIDI:
    """
    ---------------------------------
    Initialise and Choose Pygame MIDI
    ---------------------------------
    """
    midi.init()
    print_devices()
    device_id = int(input('Choose an input device [ID]:'))
    input_device = midi.Input(int(device_id))
    print('Connecting to ' + str(midi.get_device_info(device_id)) + '...')

"""
-----------------
Initialise Pygame
-----------------
"""
pygame.init()
screen = pygame.display.set_mode((700, 700), RESIZABLE)
pygame.display.set_caption('Chords AI')

font = pygame.font.Font("C:\Windows\Fonts\segoeprb.ttf", 25)
text = font.render('UNKNOWN', True, BLACK)
image = pygame.image.load(
    'IASC_1_A\\notes\\piano.png')

pygame.mixer.init()
pygame.mixer.set_num_channels(NOTES_PER_OCTAVE)  # Number of notes
channels = []
for i in range(NOTES_PER_OCTAVE):
    channels.append(pygame.mixer.Channel(i))

print('Ready! Play something...')

"""
print('PLAYED:    C')
print('PREDICTED: ' +
        to_chord(list(model.predict([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]).round()[0])))
print('PLAYED:    Fm')
print('PREDICTED: ' +
        to_chord(list(model.predict([[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]]).round()[0])))
print('PLAYED:    Unknown')
print('PREDICTED: ' +
        to_chord(list(model.predict([[1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]]).round()[0])))
"""

"""
-------------------------
Listen to Chords on Piano
-------------------------
"""
current_chord = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
while True:
    screen.fill(WHITE)
    screen.blit(text, (24, 0))
    screen.blit(image, (0, 48))

    for eve in pygame.event.get():
        if eve.type == pygame.QUIT:
            if USING_MIDI:
                input_device.close()
                midi.quit()
            pygame.quit()
        elif not USING_MIDI:
            if eve.type == pygame.KEYUP:
                note = event_to_chord(eve)
                if (note >= 0):
                    current_chord[note] = 0
            elif eve.type == pygame.KEYDOWN:
                note = event_to_chord(eve)
                if (note >= 0):
                    current_chord[note] = 1
                    channels[note].play(pygame.mixer.Sound(
                        'IASC_1_A\\notes\\' + str(NOTES[note])))

    if USING_MIDI and input_device.poll():
        event = input_device.read(1)[0]
        data = event[0]
        note = data[1]
        velocity = data[2]
        if velocity == 0:
            current_chord[note % NOTES_PER_OCTAVE] = 0
        else:
            current_chord[note % NOTES_PER_OCTAVE] = 1

    output = list(model.predict([current_chord]).round()[0])
    text = font.render(to_chord(output), True, BLACK)
    pygame.display.update()
