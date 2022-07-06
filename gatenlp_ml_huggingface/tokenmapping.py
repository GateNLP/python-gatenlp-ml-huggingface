"""
Helper functions for mapping words and per-word labels to transformers tokens and
transformers labels and back.
"""
from typing import List, Union, Optional, Dict, Callable
from collections import Counter
from transformers import BatchEncoding


def map_labels_words2tokens(
        tokresult: BatchEncoding,
        wordlabels: List[List[Union[int, str]]],
        subseq_label: Optional[Union[str, Dict, Callable]] = None,
) -> List[List[Optional[Union[int, str]]]]:
    """
    This creates the batch of label ids or labels per transformers token required for training.
    The function returns a list of list of whatever element type was passed in wordlabels.
    This return value should be usable as the "labels" field to pass for training.

    :param tokresult: the result returned from the tokenizer (should be a fast tokenizer),
        for batch of sequences of words. IMPORTANT: we expect the tokenizer to receive a list
        of pre-split words for each batch element and that tokenization was performed using
        parameter is_split_into_words=True.
    :param wordlabels: a list of list of either label ids (int) or labels (str) with the same
        number of rows and words in each row as the original batch of words passed to the tokenizer.
    :param subseq_label: This can be used to influence which label should get used for subsequent
        transformers tokens, i.e. the second and following token if a word gets mapped to more than
        one token. This can be None (use the same as for the first token), a string or int (use that
        label/code), a dict mapping from the label of the word to some other label, or a lambda
        which receives the original code/label and the how manyth token this is.

    :return:
    """
    """
    For the tokenizer result tokresult for some batch of words, create the tokelabels from the
    corresponding batch of labels. This returns a list of list with the same dimensions as
    tokresult["input_ids"]. Tokresult must have been created with
    """
    have_ovm = tokresult.get("overflow_to_sample_mapping") is not None
    lbls_batch = []
    curinputrow = -1
    curword = None
    curwordidx = 0
    for rowid in range(len(tokresult["input_ids"])):
        lbls = []
        if have_ovm:
            inputrow = tokresult["overflow_to_sample_mapping"][rowid]
        else:
            inputrow = rowid
        if curinputrow != inputrow:
            curword = None
            curinputrow = inputrow
        wordids = tokresult.word_ids(rowid)
        for wordid in wordids:
            if wordid is None:
                app = None
            elif wordid != curword:
                curword = wordid
                curwordidx = 0
                # we have the first or only token of a new word
                app = wordlabels[inputrow][wordid]
            else:
                # another token for the same word
                app = wordlabels[inputrow][wordid]
                curwordidx += 1
                if callable(subseq_label):
                    app = subseq_label(app, curwordidx)
                elif isinstance(subseq_label, dict):
                    app = subseq_label[app]
                else:
                    app = subseq_label
            lbls.append(app)
        lbls_batch.append(lbls)
    return lbls_batch


def map_labels_tokens2words(
        words: List[List[str]],
        tokresult, preds
) -> List[List[List[Union[str, int]]]]:
    """
    Generate the list of transformers-labels/labelids for each of the original words.
    This expects a list of list of words and will return a corresponding list of list of list of
    label ids or strings.

    :param words: the original batch of pre-split words that was passed to the tokenizer.
    :param tokresult: the result that was returned by the tokenizer
    :param preds: the predictions i.e. label ids or labels for each of the transformers tokens
        present in the tokeresult (including the special tokens!)
    :return: a list of list of list of label ids or labels
    """
    # first of all, create a copy of the input words structure, but with an empty list per
    # word. We always expect a list of lists of str
    labels = [[[] for _ in row] for row in words]
    have_ovm = tokresult.get("overflow_to_sample_mapping") is not None
    for rowid in range(len(tokresult["input_ids"])):
        if have_ovm:
            inputrow = tokresult["overflow_to_sample_mapping"][rowid]
        else:
            inputrow = rowid
        wordids = tokresult.word_ids(rowid)
        for tftokidx, wordid in enumerate(wordids):
            if wordid is None:
                pass  # ignore, this is for a special token
            else:
                # append the prediction for that transformer token to the word
                labels[inputrow][wordid].append(preds[rowid][tftokidx])
    return labels


def pick_mostfreq(
        per_word_label_lists: List[List[List[Union[int, str]]]]
) ->  List[List[Union[int, str]]]:
    """
    Pick a single label/label id from each list: pick the (first) most frequent value

    :param per_word_label_lists: the list of list of list of per-transformers token label predictions.
    :return: a list of list of per word labels or label ids
    """
    return [[Counter(col).most_common(1)[0][0] for col in row] for row in per_word_label_lists]


def pick_frist(
        per_word_label_lists: List[List[List[Union[int, str]]]]
) ->  List[List[Union[int, str]]]:
    """
    Pick a single label/label id from each list: pick the first value

    :param per_word_label_lists: the list of list of list of per-transformers token label predictions.
    :return: a list of list of per word labels or label ids
    """
    return [[col[0] for col in row] for row in per_word_label_lists]