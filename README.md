# Autodidact

## Introduction
This project aims to create a domain specific search engine for areas I'm interested in (neuroscience and ML, to start with). 
It's a meta-autodidactical project: I'm doing it to learn, but the project itself aims to facilitate faster learning!

## Motivation

Traditional search is great for getting high level answers to queries. With knowledge graph improvements, for example, we get a decent answers to most high-level questions using
traditional search. But search based knowledge graphs typically rely on Wikipedia, limiting the breadth and depth of results. Information locked away in the constant stream of
unstructured data found in academic pdfs, for example, isn't indexed. In many cases, this means research relies on "Googling" and looking for relevant titles, or relying on hints
from useful Tweeters. Wouldn't it be nice if there were a way to efficiently find needle's in haystacks of unstructured pdfs? Luckily, two
recent advances make this a tractable problem, even for individual developers like myself.

## Advance 1: Unsupervised domain adaptation

The growth of open source Large Language Models has made it feasible to build
semantic search without needing huge resources. In the simplest case, we can index documents using something like [MS-MARCO](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5), thus leveraging
tons of information about the structure of search without needing our own search data and without expensive model training. But this has issues:
 - queries by Bing users are generally not specific to the domain we're interested in, which is to say that the training set isn't representative.
 - The model knows nothing about what happened after it was trained (2018 in this case), so it could get confused about new information (e.g. covid-19).

Ideally, we'd be able to adapt these models to the domain we're interested in, hopefully in an unsupervised manner that leverages the domain specific data we have access to without needing
expensive labelling. Luckily, [unsupervised domain adaptation](https://www.youtube.com/watch?v=qzQPbIcQu9Q&ab_channel=OpenSourceConnections) is a rapidly advancing field. These
advances are part of the inspiration for this project.

## Advance 2: Parsing unstructured text data using ML.

ML advances has also simplified the process of parsing unstructured text [from pdfs](https://github.com/Layout-Parser/layout-parser). As luck would have it, this is especially
true for academia as there are model zoo's of parsers for academic papers. But even without this, we can fine tune the models for our needs without much fuss.

## Combining advances

Combining advances 1. and 2. I hope to build a search facilitator that helps me out. After that, we'll see where it goes. At the moment, this repo is mainly for my own learning.

## Installation

## Deep Learning dependencies (Detectron).

This library depends on detectron2 models for document AI. Getting PyTorch working with a Poetry environment can be slightly painful depending on your CUDA runtime (see [here](https://github.com/python-poetry/poetry/issues/2543)). The best (temporary) solution I found for the CUDA version on my laptop (11.3) was to use a nice task runner for poetry [poethepoet](https://github.com/nat-n/poethepoet). You can install the cli for poe and then run the following commands in the root of the repo if you have the same GPU setup as me, but if not it's a simple matter of getting the right wheel for your card and changing a few commands.

```
poetry install
poe force-cuda11
poe detectron-2
```

## Get up and running.

To download some example papers, run the following script. It will download a bunch of papers; you can edit the papers downloaded at your convenience.

```scripts/fep_pypaperbot_example.sh```

For local development, run the docker-compose stack.

```docker-compose up -d```

Access postgres locally via the command-line or use a helper tool such as pgadmin or beekper (linux users):

```psql -h localhost -U username -p 5432 -d default_database```
