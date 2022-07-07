"""
Module for creating training data from GateNLP documents
"""
import os.path
from typing import Union, IO, Optional, List, Dict
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from datasets.arrow_writer import ArrowWriter
from gatenlp import Document
from gatenlp.chunking import doc_to_ibo
from gatenlp.corpora.base import DocumentDestination


class HfTokenClassificationDestination(DocumentDestination):
    def __init__(
            self,
            outdir: str,
            annset_name: str = "",
            sentence_type: Optional[str] = None,
            token_type: str = "Token",
            token_feature: Optional[str] = None,
            chunk_annset_name: Optional[str] = None,
            chunk_types: Optional[List[str]] = None,
            type2code: Optional[Dict] = None,
            scheme: str = "BIO",
            writer_batch_size: int = 100,
            labels: Optional[List[str]] = None,
    ):
        super().__init__()
        if not os.path.isdir(outdir):
            raise Exception("Need to specify an existing directory!")
        self.outdir = outdir
        if not chunk_types or not isinstance(chunk_types, list):
            raise Exception("The parameter chunk_types must be a non-empty list of chunk type names")
        if not type2code:
            type2code = {}
        self.annset_name = annset_name
        self.sentence_type = sentence_type
        self.token_type = token_type
        self.token_feature = token_feature
        self.chunk_annset_name = chunk_annset_name
        self.chunk_types = chunk_types
        self.type2code = type2code
        self.scheme = scheme
        if labels is None:
            if scheme in ["BIO", "IOB"]:
                labels = ["O"]
                for ct in chunk_types:
                    ctn = type2code.get(ct, ct)
                    labels.append("B-"+ctn)
                    labels.append("I-" + ctn)
            else:
                raise Exception("Cannot auto-generate the labels from the chunk types and scheme")
        self.labels = labels
        self.features = Features(dict(
            id=Value(dtype="string"),
            tokens=Sequence(feature=Value(dtype="string")),
            labels=Sequence(feature=ClassLabel(
                names=labels)),
        ))
        self.writer = ArrowWriter(
            path=os.path.join(self.outdir, "tmp.arrow"),
            writer_batch_size=writer_batch_size,
            features=self.features
        )
        self.dataset = None

    def append(self, doc):
        """
        Append a document to the destination.

        Args:
            doc: the document, if None, no action is performed.
        """
        if doc is None:
            return
        assert isinstance(doc, Document)
        for n_sent, sentence_cols in enumerate(doc_to_ibo(
                doc,
                annset_name=self.annset_name,
                sentence_type=self.sentence_type,
                token_type=self.token_type,
                token_feature=self.token_feature,
                chunk_annset_name=self.chunk_annset_name,
                chunk_types=self.chunk_types,
                type2code=self.type2code,
                scheme=self.scheme,
                return_rows=False,
        )):
            data = dict(
                id=f"{self._n}-{n_sent}",
                tokens=sentence_cols[0],
                labels=sentence_cols[1]
            )
            self.writer.write(data)
        self._n += 1

    def close(self):
        self.writer.finalize()
        self.writer.close()
        self.dataset = Dataset.from_file(os.path.join(self.outdir, "tmp.arrow"))
        self.dataset.save_to_disk(self.outdir)
        os.remove(os.path.join(self.outdir, "tmp.arrow"))

