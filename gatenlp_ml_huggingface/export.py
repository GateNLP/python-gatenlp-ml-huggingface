"""
Module for creating training data from GateNLP documents
"""
import os.path
import os
import argparse
import logging
import json
from gatenlp.corpora.dirs import DirFilesSource, DirFilesCorpus
from gatenlp.utils import init_logger
from typing import Union, IO, Optional, List, Dict
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from datasets.arrow_writer import ArrowWriter
from gatenlp import Document
from gatenlp.chunking import doc_to_ibo
from gatenlp.corpora.base import DocumentDestination

## TODO: (later) the initial writers should optionally allow to gather all label strings and
##     store label info (and perhaps other info) in a second file, which can be used
##     by the training script!

## TODO: for now just support
##    * ChunkClassification: entity types, scheme, type2code, [labels]
##    * TokenClassification: label_feature, labels
##    * TextClassification: label_feature (for sequence ann if given, or document if not), labels


class HfTextClassificationDestination(DocumentDestination):
    """
    Create a HF dataset for text classification.
    """
    def __init__(
            self,
            outdir: str,
            annset_name: str = "",
            text_type: Optional[str] = None,
            text_feature: Optional[str] = None,
            label_feature: str = "class",
            writer_batch_size: int = 100,
            labels: List[str] = None,
    ):
        """
        Initialize the destination for creating a HF dataset for training text classification.
        This takes either the text of a whole document and the label from a document feature, or
        the text from an annotation (either the covered document text or a feature value) and the label
        from a feature of the annotation. If the text or the label are empty/None, the instance is
        silently ignored. If the label is not in the list of allowed labels, an exception is raised.

        :param outdir: the directory representing the HF dataset
        :param annset_name: if annotations are used, the annotation set name
        :param text_type: if this is None, no annotaitons are used and the document text is used instead.
            Otherwise the type of annotations from which to take the text and label.
        :param text_feature: if this is not None, the name of a feature which must contain the text to
            use instead of the document text (covered by the annotation). If the feature is missing/None,
            the annotation is skipped.
        :param label_feature: the feature that contains the classification label (either document feature
            or annotation feature) (required, default is "class")
        :param writer_batch_size: batch size to use for the ArrayWriter
        :param labels: a list of 2 or more classification labels or None to use whatever labels
            occur
        """
        super().__init__()
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir
        self.annset_name = annset_name
        self.text_type = text_type
        self.text_feature = text_feature
        self.labels = labels
        self.seen_labels = set()
        if label_feature is None:
            label_feature = "class"
        self.label_feature = label_feature
        self.outfh = open(os.path.join(self.outdir, "tmp.json"), "wt", encoding="utf-8")
        self.dataset = None
        self.features = None

    def append(self, doc):
        """
        Append a document to the destination.

        Args:
            doc: the document, if None, no action is performed.
        """
        if doc is None:
            return
        assert isinstance(doc, Document)
        if self.text_type is None:
            if self.text_feature:
                txt = doc.features.get(self.text_feature)
                if not txt:
                    return
            else:
                txt = doc.text
            if not txt:
                return
            label = doc.features.get(self.label_feature)
            if label is None:
                return
            if self.labels and label not in self.labels:
                raise Exception(f"Unknown label {label}")
            self.seen_labels.add(label)
            self.outfh.write(json.dumps(dict(text=txt, label=label)))
            self.outfh.write("\n")
        else:
            anns = doc.annset(self.annset_name).with_type(self.text_type)
            for ann in anns:
                if self.text_feature is None:
                    txt = doc[ann]
                else:
                    txt = ann.features.get(self.text_feature)
                    if not txt:
                        continue
                label = ann.features.get(self.label_feature)
                if label is None:
                    continue
                if self.labels and label not in self.labels:
                    raise Exception(f"Unknown label {label}")
                self.outfh.write(json.dumps(dict(text=txt, label=label)))
                self.outfh.write("\n")
                self.seen_labels.add(label)
        self._n += 1

    def close(self):
        if self._n == 0:
            raise Exception("No valid documents exported, cannot create dataset")
        if self.labels:
            labels = self.labels
        else:
            labels = list(self.seen_labels)
        self.features = Features(dict(
            # id=Value(dtype="string"),
            text=Value(dtype="string"),
            label=ClassLabel(names=labels),
        ))
        self.outfh.close()
        with open(os.path.join(self.outdir, "gatenlp_info.json"), "wt") as outfp:
            json.dump(dict(labels=labels), outfp)
        self.dataset = Dataset.from_json(os.path.join(self.outdir, "tmp.json"), features=self.features)
        self.dataset.save_to_disk(self.outdir)
        os.remove(os.path.join(self.outdir, "tmp.json"))


class HfChunkClassificationDestination(DocumentDestination):
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


def add_args_text(argparser):
    """Add argument parser arguments for text classification"""

    argparser.add_argument("--text_feature", type=str, default=None,
                           help="Annotation/document feature containing text (None)")
    argparser.add_argument("--label_feature", type=str, default="class",
                           help="Label annotation/document feature (class)")
    argparser.add_argument("--labels", type=str, default=None,
                           help="Comma separated list of possible labels (None, required)")


def add_args_chunk(argparser):
    """Add argument parser arguments for chunk classification"""
    argparser.add_argument("--token_type", type=str, default="Token",
                           help="Token annotation type (None, use document text)")
    argparser.add_argument("--token_feature", type=str, default=None,
                           help="Token feature (None, use covered document text)")
    argparser.add_argument("--chunk_types", nargs="*",
                           help="Annotation types of entity/chunk annotations")
    argparser.add_argument("--chunk_annset_name", type=str,
                           help="If specified, a different annotation set names for getting the chunk annotations")
    argparser.add_argument("--scheme", type=str, choices=["IOB", "BIO", "IOBES", "BILOU", "BMEOW", "BMEWO"],
                           default="BIO",
                           help="Chunk coding scheme to use (BIO)")


def add_args_token(argparser):
    """Add argument parser arguments for token classification"""
    argparser.add_argument("--token_type", type=str, default="Token",
                           help="Token annotation type (None, use document text)")
    argparser.add_argument("--token_feature", type=str, default=None,
                           help="Token feature (None, use covered document text)")
    argparser.add_argument("--labels", nargs="+", type=str, default=None,
                           help="Comma separated list of possible labels (None, required)")
    argparser.add_argument("--label_feature", type=str, default="class",
                           help="Label annotation/document feature (class)")


def build_argparser(description="Export training data from a directory of documents"):
    argparser = argparse.ArgumentParser(
        description=description,
    )
    subparsers = argparser.add_subparsers(
        title="Learning task",
        description="The learning task for which to run the command",
        dest="taskname",
        required=True,
    )
    argparser.add_argument("docdir", type=str,
                           help="Input directory"
                           )
    argparser.add_argument("outdir", type=str,
                           help="A directory where the output files are stored")
    argparser.add_argument("--split", type=str, default="train",
                           help="The split name (train)")
    argparser.add_argument("--recursive", action="store_true",
                           help="If specified, process all matching documents in the directory tree")
    argparser.add_argument("--exts", nargs="+", default=["bdocjs"],
                           help="File extensions to process (.bdocjs)")
    argparser.add_argument("--on_error", choices=["exit", "log", "ignore"], default="exit",
                           help="What to do if an error occurs writing the data for a document")
    argparser.add_argument("--fmt", type=str, default=None,
                           help="File format to expect for the matching documents (None: infer from extension)")
    argparser.add_argument("--annset_name", type=str, default="",
                           help="Annotation set name to use (default annotation set)")
    argparser.add_argument("--covering_type", type=str,
                           help="Type of annotations covering the text/tokens (default: None, use whole document)")
    argparser.add_argument("--debug", action="store_true",
                           help="Enable debugging mode/logging")
    add_args_text(subparsers.add_parser("text", help="Text classification"))
    add_args_chunk(subparsers.add_parser("token", help="Token classification"))
    add_args_token(subparsers.add_parser("chunk", help="Chunk classification (NER etc)"))
    return argparser


def docs2dataset_text(args, dirsrc, logger=None):
    """
    Export directory corpus to a HF text classification dataset
    """
    if args.labels:
        labels = args.labels.split(",")
    else:
        labels = None
    with HfTextClassificationDestination(
        outdir=args.outdir,
        annset_name=args.annset_name,
        text_type=args.covering_type,
        text_feature=args.text_feature,
        label_feature=args.label_feature,
        labels=labels,
    ) as dest:
        n_errors = 0
        n_read = 0
        for doc in dirsrc:
            n_read += 1
            try:
                dest.append(doc)
            except Exception as ex:
                n_errors += 1
                if args.on_error == "exit":
                    if logger:
                        logger.error(f"Problem writing document {n_read} to the destination", ex)
                    raise ex
                elif args.on_error == "log":
                    if logger:
                        logger.error(f"Problem {n_errors} writing document {n_read} to the destination", ex)
                else:
                    pass   # ignore the error
    if logger:
        logger.info(f"Number of documents read: {n_read}")
        logger.info(f"Number of errors: {n_errors}")
    pass


def docs2dataset_chunk(args, dirsrc, logger=None):
    """
    Export directory corpus to a HF text classification dataset
    """
    with HfChunkClassificationDestination(
        outdir=args.outdir,
        annset_name=args.annset_name,
        sentence_type=args.covering_type,
        token_type=args.token_type,
        token_feature=args.token_feature,
        chunk_annset_name=args.chunk_annset_name,
        chunk_types=args.chunk_types,
        scheme=args.scheme,
    ) as dest:
        n_errors = 0
        n_read = 0
        for doc in dirsrc:
            n_read += 1
            try:
                dest.append(doc)
            except Exception as ex:
                n_errors += 1
                if args.on_error == "exit":
                    if logger:
                        logger.error(f"Problem writing document {n_read} to the destination", ex)
                    raise ex
                elif args.on_error == "log":
                    if logger:
                        logger.error(f"Problem {n_errors} writing document {n_read} to the destination", ex)
                else:
                    pass   # ignore the error
    if logger:
        logger.info(f"Number of documents read: {n_read}")
        logger.info(f"Number of errors: {n_errors}")


def docs2dataset_token(args, dirsrc, logger=None):
    """
    Export directory corpus to a HF text classification dataset
    """
    pass

def run_docs2dataset():
    aparser = build_argparser()

    args = aparser.parse_args()
    if args.debug:
        logger = init_logger(lvl=logging.DEBUG)
    else:
        logger = init_logger()
    src = DirFilesSource(dirpath=args.docdir, recursive=args.recursive, fmt=args.fmt, exts=args.exts)
    if args.taskname == "text":
        docs2dataset_text(args, src, logger=logger)
    elif args.taskname == "chunk":
        docs2dataset_chunk(args, src, logger=logger)
    elif args.taskname == "token":
        docs2dataset_token(args, src, logger=logger)


if __name__ == "__main__":
    run_docs2dataset()
