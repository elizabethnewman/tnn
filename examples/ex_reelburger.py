
idx = lambda i: i - 1

x = 18 * ['_']

x[idx(1)] = 'D'
x[idx(2)] = 'P'
x[idx(3)] = 'A'
x[idx(4)] = 'N'
x[idx(5)] = 'C'
x[idx(6)] = 'F'
x[idx(7)] = 'I'
x[idx(8)] = 'E'
x[idx(9)] = 'T'
x[idx(10)] = 'Y'
x[idx(11)] = 'S'
x[idx(12)] = 'L'
x[idx(13)] = 'U'
x[idx(14)] = 'B'
x[idx(15)] = 'O'
x[idx(16)] = 'H'
x[idx(17)] = 'W'
x[idx(18)] = 'G'


def word(l):
    return " ".join([x[idx(i)] for i in l]) + "\n"

word_keys = [
    [7,4],
    [9,16,8],
    [16,7,18,16,17,3,10],
    [11,5,8,4,8],
    [17,16,3,9],
    [1,15,8,11],
    [9,16,8],
    [12,7,9,9,12,8],
    [14,15,10],
    [11,3,10],
    [17,16,8,4],
    [16,8],
    [11,8,8,11],
    [9,16,8],
    [13,6,15,11],
    [6,12,10],
    [2,3,11,9]
]
words = [str(k + 1) + ':  ' + word(i) for k, i in enumerate(word_keys)]

print(''.join(words))


