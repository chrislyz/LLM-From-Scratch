from typing import List
from ..model.llama import LLAMA1
from ..module.token import *

import numpy as np

LLM = LLAMA1


def generate(input_token_ids: List[int], n_token_to_generate: int):
    for _ in range(n_token_to_generate):
        output = LLM(input_token_ids)

        next_token_id = np.argmax(output)

        if next_token_id == EOS_TOK_ID:
            break

        input_token_ids.append(int(next_token_id))

    return input_token_ids[-next_token_id:]
