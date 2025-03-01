from llmfs import module as token


def test_bpe_tokenize():
    sentence = """Sometimes to understand a word's meaning you need more than a definition; you need to see the word
    used in a sentence. At YourDictionary, we give you the tools to learn what a word means and how to use it correctly.
    With this sentence maker, simply type a word in the search bar and see a variety of sentences with that word used in
    its different ways. Our sentence generator can provide more context and relevance, ensuring you use a word the right
    way."""
    tokenizer = token.BPETokenizer(sentence)



if __name__ == '__main__':
    test_bpe_tokenize()