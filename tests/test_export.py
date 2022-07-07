"""
Tests for the export functions
"""
import os
import shutil
from gatenlp import Document
from gatenlp_ml_huggingface.export import HfTokenClassificationDestination
from datasets import Dataset

docdir = os.path.join(os.path.dirname(__file__), "docs")


def load_docs():
    docs = []
    docs.append(Document.load(os.path.join(docdir, "testdoc_ner01.bdocjs")))
    docs.append(Document.load(os.path.join(docdir, "testdoc_ner02.bdocjs")))
    return docs


def test_export_01():
    docs = load_docs()
    dirname = "tmp_test_ds01"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with HfTokenClassificationDestination(dirname, chunk_types=["GPE", "PERSON", "ORG"]) as dest:
        for doc in docs:
            dest.append(doc)
    ds = Dataset.load_from_disk(dirname)
    print(ds)
    exs = []
    for ex in ds:
        exs.append(ex)
        print("type=", type(ex), "ex=", print(ex))
    ntok0 = len(docs[0].annset().with_type("Token"))
    ntok1 = len(docs[1].annset().with_type("Token"))
    assert ntok0 == len(exs[0]["tokens"])
    assert ntok0 == len(exs[0]["labels"])
    assert ntok1 == len(exs[1]["tokens"])
    assert ntok1 == len(exs[1]["labels"])
    shutil.rmtree(dirname)


if __name__ == "__main__":
    test_export_01()