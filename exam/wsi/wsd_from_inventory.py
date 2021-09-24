import argparse
import warnings
import json
from nltk import word_tokenize

from xml.etree import ElementTree
from egvi_gensim import WSD

warnings.filterwarnings("ignore", category=DeprecationWarning)

def generate_semeval(senseval_path):
    test_tree = ElementTree.parse(senseval_path)
    for word in test_tree.getroot():
        for inst in word.getchildren():

            inst_id = inst.attrib['id']
            context = inst.find("context")
            before, target, after = list(context.itertext())
            before = word_tokenize(before.strip())
            target = target.strip()
            after = word_tokenize(after.strip())
            yield before + [target] + after, len(before), inst_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory_path", help="Path to the WSI generated inventory", default="inventories/cc.en.300.vec.gz.top200.inventory.tsv")
    parser.add_argument("--senseval_path", help="Path to the Senseval-2013 dataset", default="../provided/data/")
    args = parser.parse_args()

    inventory_path = args.inventory_path
    senseval_path = args.senseval_path  + "contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml"

    print("Loading inventory and embeddings...")
    disambiguator = WSD(inventory_path, "en")

    results = {}
    last_word = None

    print("Loading dataset...")
    for text, target_index, instance_id in generate_semeval(senseval_path):
        word = ".".join(instance_id.split(".")[0:-1])
        sense_prefix = word + ".sense."

        if word != last_word:
            print("Done with disambiguating", last_word)
            k = 0
            sense_to_sense_index = {}

        sense_scores = disambiguator.disambiguate_tokenized(text, text[target_index])
        top_sense = sense_scores[0][0]
        
        if top_sense in sense_to_sense_index:
            results[instance_id] = {sense_prefix + str(sense_to_sense_index[top_sense]): 1}
        else:
            sense_to_sense_index[top_sense] = k
            results[instance_id] = {sense_prefix + str(k): 1}
            k += 1

        last_word = word

    with open("results_egvi.key", "w", encoding="utf-8") as f:
             f.write(json.dumps(results))
        
