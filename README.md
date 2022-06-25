# Autodidact

## Warning
At the moment, this is "research code" for my own use/experimentation so some conventions are ignored for speed (there's little testing, no-CI-CD and I use good old fashioned requirements.txt instead of poetry because the are some issues with poetry for my particular cuda runtime. I will formalise as I go alone, depending on where the project goes.

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

## Get up and running.

For local development, run the docker-compose stack.

```docker-compose up -d```

Access postgres locally via the command-line or use a helper tool such as pgadmin or beekper (linux users):

```psql -h localhost -U username -p 5432 -d default_database```
