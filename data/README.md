# Dataset files

`src/pipeline.py` downloads the public UCI SMS Spam Collection into this
folder. To add Kaggle data, download the dataset yourself and place each
CSV file directly in this folder or the project root. The loader accepts a message/text column
and a label/category column; supported binary labels include `spam`/`ham`,
`scam`/`legitimate`, or `1`/`0`.

Recommended additional source: [Spam email classification by Ashfak
Yeafi](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification).
Kaggle's public dataset metadata describes `text` and `label` fields,
`Spam`/`Ham` labels, and an Apache 2.0 license. Download the dataset ZIP,
extract its CSV file, and place that CSV here. Recheck the Kaggle page's
current license and file schema when downloading; report the observed
sample count after the file is loaded.

The loader never guesses labels or creates synthetic training examples.
Malformed/unknown-label or empty-message rows are logged and excluded only
when they make up at most 1% of a CSV; otherwise loading fails. It ignores
`combined_dataset.csv` when scanning source files. Exact duplicate messages
are removed before training, with the first source retained.

`email_spam_10k.csv` is standardized from the public
[`yxzwayne/email-spam-10k`](https://huggingface.co/datasets/yxzwayne/email-spam-10k)
dataset on Hugging Face (Apache-2.0, revision
`884e4ab960c9a4f5dc6036495f02c5ae24bf6bd2`, downloaded 2026-10-26).
The upstream JSON's `text` field is used as the message and its explicit
`is_spam` value (`1`/`0`) as the label; no labels are inferred. The
standardized CSV is retained so later pipeline runs use the same source
snapshot. The upstream corpus reports 10,899 rows; the pipeline removes
exact and normalized duplicate messages across all sources before splitting.

For each Kaggle CSV, keep a record of its exact Kaggle owner/dataset slug,
download date, declared license, and any transformations. Check that its
labels genuinely identify spam/scam versus legitimate messages before
using or reporting it. A Kaggle rehost of the UCI collection is not an
independent dataset and will mostly be removed as duplicates.

The dataset CSVs are intentionally not included in source control or
distributed by this project. Refer to the dataset provider's terms before
redistributing any downloaded data.
