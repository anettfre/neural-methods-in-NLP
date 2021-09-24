import json
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import collections
import eval

def get_senses_from_tags():
    countg = open("data/count_gold.json","r")
    countp = open("data/count_pred.json","r")

    lines_g = json.load(countg)
    lines_p = json.load(countp)
    gold_dict={}
    newtags = {}
    for i in lines_g:
        newdict = {}
        lines_g[i] = sorted(lines_g[i].items(), key=lambda item: item[1], reverse=True)
        lines_p[i] = sorted(lines_p[i].items(), key=lambda item: item[1], reverse=True)
        len_g = len(lines_g[i])
        len_p = len(lines_p[i])
    
        if len_g == len_p:
            for s1, s2 in zip(lines_g[i], lines_p[i]):
                newdict[s2[0]] = s1[1]
                newtags[s1[0]] = s2[0]

        elif len_g > len_p:
            tozip = lines_g[i][:len_p]
            for s1, s2 in zip(tozip, lines_p[i]): 
                newdict[s2[0]] = s1[1]
                newtags[s1[0]] = s2[0] 
        else:
            tozip = lines_p[i][:len_g]
            for s1, s2 in zip(lines_g[i], tozip):
                newdict[s2[0]] = s1[1]
                newtags[s1[0]] = s2[0]
        gold_dict[i] = newdict
    return newtags

#print(gold_dict)
#print(lines_p)
#print(newtags)
#with open('newtags.json','w') as new:
#    json.dump(newtags,new)
def run():
    newtags = get_senses_from_tags()

    with open("data/fix_output.txt","r") as res: #"fix_output_no_lemma.txt","r") as res:
        with open("provided/SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key","r") as key:
            pred = res.readlines()
            gold = key.readlines()

    avg_ari = []
    labels_true = []
    labels_pred = []
    words_true = {}
    words_pred = {}
    cluster = {}
    fd = 1
    for i in pred: #for each word
    
        words_in_pred = i.split(' ')
        id = words_in_pred[1].split('.') 
        id = id[0]+"."+id[1]+"."+id[2]

        for j in gold:
            words_in_gold = j.split(' ')
            gold_id = words_in_gold[1]

            word_now = gold_id.split('.')[0]+"."+gold_id.split('.')[1]
            if (gold_id == id):  
                pred_sense = words_in_pred[2].split('.')[-1] 
                pred_sense = pred_sense[0]
                gold_sense = words_in_gold[2] 
 
                if gold_sense in newtags:
                    gold_sense = newtags[gold_sense]
                else:
                    newtags[gold_sense] = 16
                    gold_sense = newtags[gold_sense]

                cluster[id] = [pred_sense, gold_sense] 
                labels_true.append(gold_sense)
                labels_pred.append(pred_sense)
  
                if word_now in words_true:
                    words_true[word_now].append(gold_sense)
                    words_pred[word_now].append(pred_sense)
                else:
                    words_true[word_now] = [gold_sense]
                    words_pred[word_now] = [pred_sense]
    ari = []
    clu = np.arange(1,16)
    ari_plot = np.zeros(15)
    idx_counter=np.zeros(15)

    for word in words_pred:
        print(word)
        ariw = metrics.adjusted_rand_score(words_true[word], words_pred[word])
        ari.append(ariw)
        print(ariw)
        idx = max([int(i) for i in words_pred[word]])
        ari_plot[idx] += ariw
        idx_counter[idx] += 1
 
    for i in range(len(idx_counter)):
        if idx_counter[i] != 0 and ari_plot[i] !=0:
            ari_plot[i] = ari_plot[i]/idx_counter[i]

    egvi_plot = [0.,         -0.02815126 , 0.00238873,  0.02681942 , 0.05406738 , 0.02150336,
      0.01309762 , 0.01768448 , 0.,          0.0166902 ,  0.   ,       0.,
      0.        ,  0.        ]
    plt.plot(clu[:-1], ari_plot[:-1])
    #print(ari_plot[:-1])
    plt.plot(clu[:-1], egvi_plot)
    plt.title("average ARI for each number of cluster senses (max 14)")
    plt.ylabel("ARI")
    plt.xlabel("Number of clusters")
    #plt.show()
    plt.savefig("Clusters_ARI_lemma14.png")

    print("average ARI score over all words")
    print(sum(ari)/len(ari))


if __name__ == "__main__":
    eval.run(file="data/results.key")
    run()