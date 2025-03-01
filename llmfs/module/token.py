from typing import Final

__all__ = ["EOS_TOK_ID"]


EOS_TOK_ID: Final[int] = 0


class Preprocessor:
    def __call__(self, text: str) -> str:
        self.text = text
        # preprocess pipeline
        self._normalize()
        self._pre_tokenize()

        return self.text

    def _normalize(self):
        self.text = self.text.lower()

    def _remove_punctuation(self):
        pass

    def _pre_tokenize(self):
        pass


class BaseTokenizer:
    def __init__(self, text: str):
        p = Preprocessor()
        self.text = p(text)


class NGramTokenizer(BaseTokenizer):
    def __init__(self, text: str):
        super().__init__(text)

    def _create_ngrams(self, tokens, n: int = 1):
        ngrams = {}
        for ngram in (tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)):
            ngrams[ngram] += 1
        return ngrams


class BPETokenizer(BaseTokenizer):
    def __init__(self, text):
        super().__init__(text)
        print(self.text)

    def merge(self):
        pass


class WordPieceTokenizer(BaseTokenizer):
    pass
