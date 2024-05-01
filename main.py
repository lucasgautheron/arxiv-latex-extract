#!/usr/bin/env python
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from os import sched_getaffinity
from os.path import basename
from shutil import copy
from tempfile import TemporaryDirectory
from typing import Callable
import pandas as pd
import logging

from tqdm import tqdm

from ale import LATEX_DIR
from ale.arxiv import delete, download
from ale.cleaner import ArxivCleaner

def clean(archive, output, target_dir=LATEX_DIR, filter_func=lambda _: True, verbose=False):
    # create temporary work directory
    with TemporaryDirectory(delete=False) as work_dir:
        arxiv_cleaner = ArxivCleaner(
            data_dir=archive,
            work_dir=work_dir,
            target_dir=target_dir,
            filter_func=filter_func
        )

        return arxiv_cleaner.run(out_fname=output, verbose=verbose)

def process(archive, **kwargs):
    if isinstance(archive, Callable): # handle lazy downloading
        archive = archive()

    output = f"{basename(archive)}.jsonl"
    path = clean(archive, output, **kwargs)
    delete(archive)
    return path

articles = pd.read_parquet("../quantum-gravity/inspire-harvest/database/articles.parquet")[
    ["article_id", "categories", "arxiv"]
]
articles = articles[
    articles["categories"].map(
        lambda l: any([x in l for x in ["Astrophysics", "Phenomenology-HEP", "Theory-HEP", "Gravitation and Cosmology"]])
    )
]
articles = articles[articles["arxiv"].map(len)>0]
whitelist = articles["arxiv"].tolist()
print(len(whitelist))

def filter_func(arxiv_id):
    return arxiv_id in whitelist

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    cutoff = datetime(2000, 1, 1) # do not process papers older than 2000

    # Parallelize to make things faster
    with Pool(num_workers:=1) as p:
        print(f"Parallel processing on {num_workers} workers.")

        # save results in a tmpdir first and copy them into LATEX_DIR only when
        # processing is finished. This avoids partial files in case of errors
        with TemporaryDirectory(delete=False) as target_dir:
            tasks = list(download(lazy=True, cutoff=cutoff))
            kwargs = dict(filter_func=filter_func, target_dir=target_dir)

            for path in tqdm(p.imap_unordered(partial(process, **kwargs), tasks), total=len(tasks)):
                copy(path, LATEX_DIR)
