data = open('testDataIO.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)

print('data has %d characters, %d unique.' % (data_size, vocab_size))
print('data:', data)
print('chars:', chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

print('char_to_ix:', char_to_ix)
print('ix_to_char:', ix_to_char)
