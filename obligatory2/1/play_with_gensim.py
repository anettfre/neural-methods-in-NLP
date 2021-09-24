#!/bin/env python3
# coding: utf-8

import sys
import gensim
import logging
import zipfile
import json
import random

def load_embedding(modelfile):
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
        modelfile.endswith(".txt.gz")
        or modelfile.endswith(".txt")
        or modelfile.endswith(".vec.gz")
        or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open("meta.json")
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print("============")
            # Loading the model itself:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    # Unit-normalizing the vectors (if they aren't already):
    emb_model.init_sims(
        replace=True
    )
    return emb_model


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    embeddings_file = sys.argv[1]  # File containing word embeddings

    word_file = sys.argv[2] #File containing words to be used

    logger.info("Loading the embedding model...")
    model = load_embedding(embeddings_file)
    logger.info("Finished loading the embedding model...")

    logger.info(f"Model vocabulary size: {len(model.vocab)}")

    logger.info(f"Random example of a word in the model: {random.choice(model.index2word)}")

    with open(word_file) as f, open("words_out.txt", "w") as out:
        words = f.readlines()
        words = [i.rstrip() for i in words]
        for word in words:
            if word in model:
                print("=====")
                print("Associate\tCosine")
                out.write("Associate" + "\t" + "Cosine" + "\n")
                for i in model.most_similar(positive=[word], topn=5):
                    print(f"{i[0]}\t{i[1]:.3f}")
                    out.write(i[0] + "\t" + "{:.3f}".format(i[1]) + "\n")
                out.write("\n")
                print("=====")
            else:
                print(f"{word} is not present in the model")
                out.write(f"{word} is not present in the model" + "\n")
