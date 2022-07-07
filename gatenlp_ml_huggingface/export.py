"""
Module for creating training data from GateNLP documents
"""
import os.path
import os
import argparse
import logging
from gatenlp.corpora.dirs import DirFilesSource
from gatenlp.utils import init_logger
from typing import Union, IO, Optional, List, Dict
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from datasets.arrow_writer import ArrowWriter
from gatenlp import Document
from gatenlp.chunking import doc_to_ibo
from gatenlp.corpora.base import DocumentDestination

## TODO: for now just support
##    * ChunkClassification: entity types, scheme, type2code, [labels]
##    * TokenClassification: label_feature, labels
##    * TextClassification: label_feature (for sequence ann if given, or document if not), labels
## TODO: Use the build_argparser pattern:
##    * first build the basic argparser which has --task and common parms
##    * parse known args and get args, extra (extra not needed here as we always add args)
##    * build the basic argparser again, then call the function to add task-specific options
##    * properly parse args again with final argparser

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
    argparser.add_argument("--token_type", type=str, default=None,
                           help="Token annotation type (None, use document text)")
    argparser.add_argument("--token_feature", type=str, default=None,
                           help="Token feature (None, use covered document text)")
    argparser.add_argument("--labels", nargs="+", type=str, default=None,
                           help="List of possible labels (None, required)")


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
                           help="List of possible labels (None, required)")


def build_argparser(description="Export training data from a directory of documents"):
    argparser = argparse.ArgumentParser(
        description=description,
        add_help=False,
    )
    argparser.add_argument("docdir", type=str,
                           help="Input directory"
                           )
    argparser.add_argument("outdir", type=str,
                           help="A directory where the output files are stored")
    argparser.add_argument("--task", choices=["token", "chunk", "text"], required=True,
                           help="ML task: token, chunk, or text (required)")
    argparser.add_argument("--split", type=str, default="train",
                           help="The split name (train)")
    argparser.add_argument("--recursive", action="store_true",
                           help="If specified, process all matching documents in the directory tree")
    argparser.add_argument("--exts", nargs="+", default=[".bdocjs"],
                           help="File extensions to process (.bdocjs)")
    argparser.add_argument("--on_error", choices=["exit", "log", "ignore"], default="exit",
                           help="What to do if an error occurs writing the data for a document")
    argparser.add_argument("--fmt", type=str, default=None,
                           help="File format to expect for the matching documents (None: infer from extension)")
    argparser.add_argument("--annset_name", type=str, default="",
                           help="Annotation set name to use (default annotation set)")
    argparser.add_argument("--covering_type", type=str,
                           help="Type of annotations covering the text/tokens (default: None, use whole document)")
    argparser.add_argument("-h", "--help", action="store_true")
    argparser.add_argument("--debug", action="store_true",
                           help="Enable debugging mode/logging")
    return argparser


def run_docs2dataset(args):
    if args.debug:
        logger = init_logger(lvl=logging.DEBUG)
    else:
        logger = init_logger()
    src = DirFilesSource(dirpath=args.docdir, recursive=args.recursive, exts=args.exts, fmt=args.fmt)
    with HfChunkClassificationDestination(
        outdir=args.outdir,
        annset_name=args.annset_name,
        sentence_type=args.sentence_type,
        token_type=args.token_type,
        token_feature=args.token_feature,
        chunk_annset_name=args.chunk_annset_name,
        chunk_types=args.chunk_types,
        scheme=args.scheme,
    ) as dest:
        n_errors = 0
        n_read = 0
        for doc in src:
            n_read += 1
            try:
                dest.append(doc)
            except Exception as ex:
                n_errors += 1
                if args.on_error == "exit":
                    logger.error(f"Problem writing document {n_read} to the destination", ex)
                    raise ex
                elif args.on_error == "log":
                    logger.error(f"Problem {n_errors} writing document {n_read} to the destination", ex)
                else:
                    pass   # ignore the error
    logger.info(f"Number of documents read: {n_read}")
    logger.info(f"Number of errors: {n_errors}")


if __name__ == "__main__":
    aparser = build_argparser()
    args = aparser.parse_args()
    if args.help and args.task is None:
        aparser.print_help()
    if aparser.task == "text":
        aparser = build_argparser(
            description="Export training data for text classification from a directory of documents")
        add_args_text(aparser)
        args = aparser.parse_args()
    elif aparser.task == "chunk":
        aparser = build_argparser(
            description="Export training data for chunking from a directory of documents")
        add_args_chunk(aparser)
        args = aparser.parse_args()
    elif aparser.task == "token":
        aparser = build_argparser(
            description="Export training data for token classification from a directory of documents")
        add_args_token(aparser)
        args = aparser.parse_args()
    run_docs2dataset(args)
