from __future__ import annotations
import re


def ngram(input_str, n: int = 1, sep: str = "\\s+") -> list[str]:
    res_list = []
    input_list = re.split(sep, input_str)
    for i in range(len(input_list) - n + 1):
        res_list.append(input_list[i : i + n])
    return res_list
    

candidate = """It is a guide to action which
ensures that the military always obeys
the commands of the party."""
# test_str = "hello world"
reference = """It is a guide to action that
ensures that the military will forever
heed Party commands."""
print(ngram(candidate, 2))
print(ngram(reference, 2))

n1 = ngram(candidate, 2)
n2 = ngram(reference, 2)
print(len(n1), len(n2))
print([x for x, y in zip(n1, n2) if x == y])
