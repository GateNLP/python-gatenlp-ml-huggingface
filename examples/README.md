# Examples


## Text classification

This usually starts with a corpus of documents as a training set where one of several classes/labels has been assigned 
to the text of either the whole document text or to annotations covering all or parts of the document text.
The label for the text must be stored in a document feature if the whole document text should be used, or in the
annotation covering the text.

Note that transformer models are used, so the length of the text is limited to what the transformer model can handle.

#### Importing a dataset

This is mainly used for testing: it can be used to convert an existing Huggingface text classification dataset
to a directory corpus of GateNLP documents. For this the command `gatenlp-huggingface-dataset2docs` can be used
with the subcommand text. 

To get usage information for the `gatenlp-huggingface-dataset2docs` command use `gatenlp-huggingface-dataset2docs --help`. 
This shows the general format of the command and global options which can be used with all subcommands: 

```
usage: gatenlp-huggingface-dataset2docs [-h] [--splits SPLITS]
                                        [--annset_name ANNSET_NAME]
                                        [--covering_type COVERING_TYPE]
                                        [--log-every LOG_EVERY] [--debug]
                                        {text,token,chunk} ... dataset outdir

Import dataset to a directory of documents

positional arguments:
  dataset               The HF dataset name or a local directory with the
                        directory
  outdir                Output directory where the gatenlp documents are
                        stored

optional arguments:
  -h, --help            show this help message and exit
  --splits SPLITS       The name or names of splits to import (default: all)
  --annset_name ANNSET_NAME
                        Annotation set name to use (default annotation set)
  --covering_type COVERING_TYPE
                        Type of annotations covering the text/tokens (default:
                        None, use whole document)
  --log-every LOG_EVERY
                        Log a progress message every that many processed
                        documents (1000)
  --debug               Enable debugging mode/logging

Learning task:
  The learning task for which to run the command

  {text,token,chunk}
    text                Text classification
    token               Token classification
    chunk               Chunk classification (NER etc)
```

In order to show the options for the `text` subcommand use `gatenlp-huggingface-dataset2docs text --help`:

```
usage: gatenlp-huggingface-dataset2docs text [-h] [--text_field TEXT_FIELD]
                                             [--label_field LABEL_FIELD]
                                             [--idx_field IDX_FIELD]
                                             [--idx_feature IDX_FEATURE]
                                             [--label_feature LABEL_FEATURE]

optional arguments:
  -h, --help            show this help message and exit
  --text_field TEXT_FIELD
                        Name of the text field in the dataset (sentence)
  --label_field LABEL_FIELD
                        Name of the label/classe field in the dataset (label)
  --idx_field IDX_FIELD
                        Name of the index field in the dataset (None). If
                        none, ignore.
  --idx_feature IDX_FEATURE
                        Name of the index feature name. If none, do not store
                        index.
  --label_feature LABEL_FEATURE
                        Label annotation/document feature (class)
```

Important options:

* `--covering_type` : if this is not specified, then no annotation is created for the text and the label
    is stored as a document feature. If a type is specified, an annotation is created in annotation set specified
    with `--annset_name` that covers the document text and the label is stored in an annotation feature.
* `--text_field`: the name of the field used to store the text in the Huggingface dataset. This is often "sentence"
    (the default) but may be "text", or anything else for a specific dataset.
* `--label_field`: the name of the field used to store the label in the Huggingface dataset.
* `--label_feature`: the name of the feature used to store the label in the GateNLP document



Here is an example call to convert the `rotten_tomatoes` dataset to a directory corpus in directory `testdir-rotten-tomatoes`:

```
gatenlp-huggingface-dataset2docs text --text_field text rotten_tomatoes testdir-rotten-tomatoes --covering_type Text
```
