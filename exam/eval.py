import json
import os
import re
from collections import defaultdict
from scipy.stats import spearmanr

def get_n_senses_corr(gold_key, new_key):
    #senses_lemma_gold = defaultdict(set)
    senses_lemma_gold = defaultdict(set)
    
    #senses_lemma_sys = defaultdict(set)
    senses_lemma_sys = defaultdict(set)

    for dic, key in [(senses_lemma_gold, gold_key), (senses_lemma_sys, new_key)]:
        with open(key, 'r') as fin:
            for line in fin:
                split = line.split()
                lemma = split[0]
                senses = [x.split('/')[0] for x in split[2:]]
                #print(senses)
                dic[lemma].update(senses)

    oredered_lemmas = list(senses_lemma_gold.keys())
    #print(senses_lemma_gold)
    #print(dic)
    corr = {}
    for ending, name in [('', 'all'), ('v', 'VERB'), ('n', 'NOUN'), ('j', 'ADJ')]:
        g = [len(senses_lemma_gold[x]) for x in oredered_lemmas if x.endswith(ending)]
        print(g)
        n = [len(senses_lemma_sys[x]) for x in oredered_lemmas if x.endswith(ending)]
        print(n)
        c = spearmanr(g, n)
        corr[name] = c.correlation, c.pvalue
    return corr


def fix_output(labeling):
    with open("data/fix_output.txt", 'w', encoding='utf-8') as fout:
        lines = []
        for instance_id, clusters_dict in labeling.items():
            clusters = sorted(clusters_dict.items(), key=lambda x: x[1])
            clusters_str = ' '.join([('%s/%d' % (cluster_name, count)) for cluster_name, count in clusters])
            lemma_pos = instance_id.rsplit('.', 1)[0]
            lines.append('%s %s %s' % (lemma_pos, instance_id, clusters_str))
        fout.write('\n'.join(lines))
        fout.flush()
        gold_key_path = "provided/SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key"
        #scores = get_scores(gold_key_path, fout.name)
        """
        if key_path:
            logging.info('writing key to file %s' % key_path)
            with open(key_path, 'w', encoding="utf-8") as fout2:
                fout2.write('\n'.join(lines))
        """
        correlation = get_n_senses_corr(gold_key_path, fout.name)

    return correlation
#def get_stuff(gold, pred):
#

def run(file="data/results.key"):
    #Read data
    results = {}
    #for lemmatization:
    #with open("results.key", 'r', encoding='utf-8') as f:
    #    results = json.load(f)
    #for no lemmation:
    import json
    with open(file,"r") as f:
        results = json.load(f)

    gold = []
    with open("provided/SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key", 'r', encoding='utf-8') as f:
        data = f.readlines()
        for i in data:
            gold.append(i)

    #"image.n.73": {"image.n.sense.1": 15}, "image.n.35": {"image.n.sense.4": 14, "image.n.sense.2": 1},
    #Get the most likely sense if there are more than one guess
    #Else choose the only one (clean up brackets from dict)
    for i in results:
        if len(results[i]) > 1:
            #print(results[i])
            max_key = max(results[i], key=results[i].get)
            results[i] = {max_key: results[i][max_key]}


    corr = fix_output(results)
    #corr = get_n_senses_corr("provided/SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key", "fix_output_no_lemma.txt")
    print(corr)
    ret = {}
    with open("data/fix_output.txt", 'r', encoding='utf-8') as f:
        res = f.readlines()
        for line in res:
            split = line.split('\t')
            word = split[0]
            # results = list(zip(columns[1:], map(float, split[1:])))
            #result = split[column]
            if word not in ret:
                ret[word] = {}
            #ret[word][metric] = float(result)




    gold = []
    pred = []
    with open("data/fix_output.txt","r") as res:
        with open("provided/SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key","r") as key:
            pred = res.readlines()
            gold = key.readlines()
    cluster = {}
    for i in pred:
        words_in_pred = i.split(' ')
        id = words_in_pred[1].split('.') #re.match(i,gold)  #('^(?=.*(i).*$'),gold)#(find i word.sentence_number (feks add.v.1))
        id = id[0]+"."+id[1]+"."+id[2]
    
        for j in gold:
            words_in_gold = j.split(' ')
            gold_id = words_in_gold[1].split('.')
            gold_id = gold_id[0]+'.'+gold_id[1]+"."+gold_id[2]
        
            if (gold_id == id):   #(find j word.sentence_number) == id:
                pred_sense = words_in_pred[2].split('.')[-1] #(find i sense)
                pred_sense = pred_sense[0]
                gold_sense = words_in_gold[2]  #re.match() #(find j sense)
                cluster[id] = [pred_sense, gold_sense] #må lagre hvilket ord det er også
    #print('-'*10)
    #print(cluster)
    #print('-'*10)
    """
    {add.v.1: {0, #4032::}, add.v.2: {1, #3381::}, …}
    """
    most_freq_pred = {}
    most_freq_gold = {}
    #Må skille på ord til nye clustere?
    first = True
    import json
    newdict={}
    countp = open("data/count_pred.json","w")
    counter = 0
    test = []
    for i in cluster:
        word = i.split('.')
        word = word[0]+"."+word[1] 
        counter += 1
        if first:
            #word = i.split('.')[0] #regex(frem til første punktum) # add
            prev_word = word
            first = False
        #else:
        #    #print("first else")
        #    #word = i.split('.')[0] #.regex(frem til første punktum)
        if word == prev_word:
            senses = cluster[i] #{0, #4032::}
            if senses[0] in most_freq_pred:#.items():
                most_freq_pred[senses[0]] += 1
            else:
                most_freq_pred[senses[0]] = 1
            if counter == len(cluster):
                newdict[prev_word] = most_freq_pred
                most_freq_pred = {}

        elif word != prev_word:
            first = True
            newdict[prev_word]=most_freq_pred
            most_freq_pred = {}
            senses = cluster[i]
            most_freq_pred[senses[0]] = 1



    json.dump(newdict, countp)
    print(counter)
    print(len(test))
    print(len(newdict))
    for i in test:
        print(newdict[i])
    newdict = {}
    countg = open("data/count_gold.json","w")
    first = True
    counter = 0
    print(len(cluster))
    for i in cluster:
        word = i.split('.')
        word = word[0]+"."+word[1]
        counter += 1
        if first:
            #word = i.split('.')[0] 
            prev_word = word
            first = False
    
        if word == prev_word:
        
            senses = cluster[i] #{0, #4032::}
            #prev_word = word
            if senses[1] in most_freq_gold:#.items():
                most_freq_gold[senses[1]] += 1
            else:
                most_freq_gold[senses[1]] = 1
            if counter == len(cluster):
                newdict[prev_word] = most_freq_gold
                most_freq_gold = {}
        
 
        elif word != prev_word:      
            first = True
            newdict[prev_word] = most_freq_gold
            most_freq_gold = {}
            senses = cluster[i]
            most_freq_gold[senses[1]] = 1


    print(counter)
    json.dump(newdict,countg)
    print(len(newdict))
    print("Done")
    countp.close()
    countg.close()


if __name__ == "__main__":
    run()