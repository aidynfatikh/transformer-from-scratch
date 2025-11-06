import pickle

# Add a padding token constant
PADDING_TOKEN = 0

def encode(string, vocab, max_length=None):
    # string to tokens
    bytes = list(string.encode("utf-8"))
    merges = len(vocab) - 256
    token = 256
    for _ in range(merges):
        bytes = merge(bytes, vocab[token], token)
        token += 1

    # Add padding if max_length is specified
    if max_length is not None:
        if len(bytes) > max_length:
            bytes = bytes[:max_length]  # Truncate if too long
        else:
            bytes.extend([PADDING_TOKEN] * (max_length - len(bytes)))  # Pad if too short

    return bytes

def decode(tokens, vocab):
    # Remove padding tokens
    tokens = [token for token in tokens if token != PADDING_TOKEN]

    # tokens to string
    tokens = b"".join(vocab[token] for token in tokens)
    text = tokens.decode("utf-8")
    return text

def get_pairs(tokens):
    # get counts of all pairs
    pairs = {}
    for pair in zip(tokens, tokens[1:]):
        pairs[pair] = pairs.get(pair, 0) + 1
    
    return pairs

def merge(tokens, pair, pair_token):
    # merge all pairs in tokens
    new_tokens = []
    i = 0

    while i < len(tokens):
        if i < len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(pair_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def build_vocab(data, vocab_size):
    # build vocab
    num_merges = vocab_size - 256
    token = 256

    tokens = list(data.encode("utf-8"))

    merges = {}
    for _ in range(num_merges):
        stats = get_pairs(tokens)

        if not stats:
            break

        pair = max(stats, key=stats.get)
        tokens = merge(tokens, pair, token)
        merges[pair] = token
        token += 1

    vocab = {idx: bytes([idx]) for idx in range(256)}
    vocab[PADDING_TOKEN] = b""  # Add padding token
    token = 256

    for (a, b), token in merges.items():
        vocab[token] = vocab[a] + vocab[b]
        token += 1

    return vocab

def save_tokenizer(vocab, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(vocab, file)

def load_tokenizer(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)
