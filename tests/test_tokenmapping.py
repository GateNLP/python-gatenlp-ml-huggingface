"""
Tests for the token mapping functions
"""
from transformers import AutoTokenizer
from gatenlp_ml_huggingface.tokenmapping import map_labels_tokens2words, map_labels_words2tokens

words1 = [["just"], ["and", "this"]]
codes1 = [["BX"  ], ["O",   "BX"]]
words2 = [["just", "this"], ["and", "this"]]
codes2 = [["BX",   "IX"],   ["O",   "BX"]]
words3 = [["highschoolgirl", "is", "almost", "super-duper-ridiculous"], ["yes", "it", "is", "me", "."]]
codes3 = [["O",              "O",  "B-X",    "I-X"],                    ["B-Y", "B-Z", "O", "O", "O"]]

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")


def test_words2tokens_01():
    assert tokenizer is not None
    ret = tokenizer(words1,
                    is_split_into_words=True,
                    stride=1,
                    return_token_type_ids=False,
                    return_overflowing_tokens=True,
                    padding=True,
                    max_length=6,  # could use tokenizer.model_max_length here
                    truncation=True,  # use the specified maxlength
                    return_offsets_mapping=False)
    lbls = map_labels_words2tokens(ret, codes1)
    print("01", lbls)
    # assert lbls == [[None, 'BX', None, None], [None, 'O', 'BX', None]]


def test_words2tokens_02():
    assert tokenizer is not None
    ret = tokenizer(words2,
                    is_split_into_words=True,
                    stride=1,
                    return_token_type_ids=False,
                    return_overflowing_tokens=True,
                    padding=True,
                    max_length=6,  # could use tokenizer.model_max_length here
                    truncation=True,  # use the specified maxlength
                    return_offsets_mapping=False)
    lbls = map_labels_words2tokens(ret, codes2)
    print("02", lbls)
    # assert lbls == [[None, 'BX', 'IX', None], [None, 'O', 'BX', None]]


def test_words2tokens_03():
    assert tokenizer is not None
    ret = tokenizer(words3,
                    is_split_into_words=True,
                    stride=1,
                    return_token_type_ids=False,
                    return_overflowing_tokens=True,
                    padding=True,
                    max_length=4,  # could use tokenizer.model_max_length here
                    truncation=True,  # use the specified maxlength
                    return_offsets_mapping=False)
    lbls = map_labels_words2tokens(ret, codes3)
    print("03", lbls)
    assert lbls == [
        [None, 'O', 'O', None],
        [None, 'O', 'O', None],
        [None, 'O', 'O', None],
        [None, 'O', 'O', None],
        [None, 'O', 'B-X', None],
        [None, 'B-X', 'I-X', None],
        [None, 'I-X', 'I-X', None],
        [None, 'I-X', 'I-X', None],
        [None, 'I-X', 'I-X', None],
        [None, 'I-X', 'I-X', None],
        [None, 'I-X', 'I-X', None],
        [None, 'B-Y', 'B-Z', None],
        [None, 'B-Z', 'O', None],
        [None, 'O', 'O', None],
        [None, 'O', 'O', None]]

