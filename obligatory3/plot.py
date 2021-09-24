from conllu import parse
from collections import Counter
import matplotlib.pyplot as plt
import math

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 7})

data = parse(open("norne-nb-in5550-train.conllu", "r").read())
labels = [data[i][0]['misc']['name'] for i in range(len(data))]

c = Counter(labels)
k = list(c.keys())
v = list(c.values())
a = sum(v)
x = [math.log(i,10) for i in v]
x = sorted(x, reverse = True)
plt.figure(figsize = (8,6))
plt.bar(c.keys(), x, width = 0.6, color = 'cornflowerblue', align = 'center')
plt.xlabel('Entity type', fontsize = 11)
plt.ylabel('Logged value (%)', fontsize = 11)
plt.title("Class distribution", fontsize = 11)
plt.savefig('class_dist.png', dpi = 300)