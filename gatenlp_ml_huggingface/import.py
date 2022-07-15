"""
Module for importing gatenlp directory corpus from a HF dataset.
"""
import os.path
import os
import argparse
import logging
from gatenlp.corpora.dirs import DirFilesSource, DirFilesDestination
from gatenlp.utils import init_logger
from datasets import load_dataset
from typing import Union, IO, Optional, List, Dict
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from datasets.arrow_writer import ArrowWriter
from gatenlp import Document
from gatenlp.chunking import doc_to_ibo
from gatenlp.corpora.base import DocumentDestination

# Usage note: this apparently needs the following order:
#  import.py {text,chunk,token} [cmd-specific-options] dataset outdir [general-options]
# Example use for textclassification:
# python gatenlp_ml_huggingface/import.py text --text_field text rotten_tomatoes tmpdir1 --covering_type Text

def add_args_text(argparser):
    """Add argument parser arguments for text classification"""

    argparser.add_argument("--text_field", type=str, default="sentence",
                           help="Name of the text field in the dataset (sentence)")
    argparser.add_argument("--label_field", type=str, default="label",
                           help="Name of the label/classe field in the dataset (label)")
    argparser.add_argument("--idx_field", type=str, default=None,
                           help="Name of the index field in the dataset (None). If none, ignore.")
    argparser.add_argument("--idx_feature", type=str, default=None,
                           help="Name of the index feature name. If none, do not store index.")
    argparser.add_argument("--label_feature", type=str, default="class",
                           help="Label annotation/document feature (class)")


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


def build_argparser(description="Import dataset to a directory of documents"):
    argparser = argparse.ArgumentParser(
        description=description,
    )
    subparsers = argparser.add_subparsers(
        title="Learning task",
        description="The learning task for which to run the command",
        dest="taskname",
        required=True,
    )
    add_args_text(subparsers.add_parser("text", help="Text classification"))
    add_args_chunk(subparsers.add_parser("token", help="Token classification"))
    add_args_token(subparsers.add_parser("chunk", help="Chunk classification (NER etc)"))
    argparser.add_argument("dataset", type=str,
                           help="The HF dataset name or a local directory with the directory")
    argparser.add_argument("outdir", type=str,
                           help="Output directory where the gatenlp documents are stored"
                           )
    argparser.add_argument("--splits", type=str,
                           help="The name or names of splits to import (default: all)")
    argparser.add_argument("--annset_name", type=str, default="",
                           help="Annotation set name to use (default annotation set)")
    argparser.add_argument("--covering_type", type=str,
                           help="Type of annotations covering the text/tokens (default: None, use whole document)")
    argparser.add_argument("--debug", action="store_true",
                           help="Enable debugging mode/logging")
    return argparser


def dataset2docs_text(args, dest, logger=None):
    """
    Import text classification dataset to directory.
    """
    dsplits = load_dataset(args.dataset)
    if args.splits is None:
        splits = list(dsplits.keys())
    n_written = 0
    for split in splits:
        ds = dsplits[split]
        print("DEBUG RUNNING FOR SPLIT", split)
        for ex in ds:
            print("DEBUG: ", ex)
            txt = ex[args.text_field]
            lbl = ex[args.label_field]
            idx = None
            if args.idx_field and args.idx_feature:
                idx = ex[args.idx_field]
            doc = Document(txt)
            if args.covering_type:
                ann = doc.annset(args.annset_name).add(0, len(txt), args.covering_type)
                ann.features[args.label_feature] = str(lbl)
                if args.idx_field and args.idx_feature:
                    ann.features[args.idx_feature] = idx
            else:
                doc.features[args.label_feature] = str(lbl)
                if args.idx_field and args.idx_feature:
                    doc.features[args.idx_feature] = idx
            dest.append(doc)
            n_written += 1
    if logger:
        logger.info(f"Number of documents written: {n_written}")


def dataset2docs_chunk(args, dest, logger=None):
    """
    Import a chunking dataset.
    """
    raise Exception("Not implemented yet")


def dataset2docs_token(args, dirsrc, logger=None):
    """
    Import a token classification dataset
    """
    raise Exception("Not yet implemented")

def run_dataset2docs():
    aparser = build_argparser()

    args = aparser.parse_args()
    if args.debug:
        logger = init_logger(lvl=logging.DEBUG)
    else:
        logger = init_logger()
    dest = DirFilesDestination(dirpath=args.outdir, fmt="bdocjs", ext="bdocjs")
    if args.taskname == "text":
        dataset2docs_text(args, dest, logger=logger)
    elif args.taskname == "chunk":
        dataset2docs_chunk(args, dest, logger=logger)
    elif args.taskname == "token":
        dataset2docs_token(args, dest, logger=logger)


if __name__ == "__main__":
    run_dataset2docs()
