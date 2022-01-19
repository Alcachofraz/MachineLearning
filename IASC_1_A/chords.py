from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

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

CHORD_NUM = 24
NOTES_NUM = 12
NOTES_PER_CHORD = 3

MAJOR_CHORD_PATTERN = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
MINOR_CHORD_PATTERN = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

OUTPUT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

TOTAL_DATA_LENGTH = 10000
FORCED_CHORDS_NUM = 100 # Per chord
RANDOM_CHORDS_NUM = TOTAL_DATA_LENGTH - (FORCED_CHORDS_NUM * CHORD_NUM)

"""
This means in 10000 chords, 2400 will be granted meaningful chords
and 7600 will be random chords.
"""

# Takes an array [list] (octave) and rotates its positions right [n] times.
def rotate_right(octave, n):
    return (octave[len(octave) - n:len(octave)]
                 + octave[0:len(octave) - n])

# Takes an array [list] (octave) and rotates its positions right [n] times.
def rotate_left(octave, n):
    return (octave[len(octave) - n:len(octave)]
                 + octave[0:len(octave) - n])

# Takes an array [list] (octave) and returns a 24 length list indicating
# which chord is being played, if any.
def is_chord(octave):
    ret = OUTPUT.copy()
    if octave.count(1) != 2:
        return ret
    for i in range(NOTES_NUM):
        temp = rotate_right(octave, i)
        if temp[0] == 1 and temp[7] == 1:
            if temp[3] == 1:
                ret[]
                return []
            elif temp[4] == 1:
            else:


X = np.array([], dtype=int)
Y = np.array([], dtype=int)

for i in range(NOTES_NUM):
    X = np.append(rotate_right(MAJOR_CHORD_PATTERN, i))
    Y = np.append()

Y = np.reshape(Y, (TOTAL_DATA_LENGTH, 2))

regr = MLPRegressor(hidden_layer_sizes=(16, 8),
    activation = 'tanh',
    solver = 'adam',
    max_iter = 10000,
    verbose = False)

model = regr.fit(X, Y)

TEST = np.concatenate(([A], [B], random_pattern(16)))
print(TEST)
print(model.predict(TEST).round())
