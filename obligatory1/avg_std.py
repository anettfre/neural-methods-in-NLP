import statistics

def average(lst): 
    return sum(lst) / len(lst)

def stdevoflist(lst):
	return statistics.stdev(lst)

acc = [0.5056, 0.5091, 0.5009]
pre = [0.4348, 0.4365, 0.4321]
rec = [0.4325, 0.4280, 0.4249]
f1 = [0.4277, 0.4307, 0.4263]

av_acc = average(acc)
av_pre = average(pre)
av_rec = average(rec)
av_f1 = average(f1)

print(av_acc, av_pre, av_rec, av_f1)

st_acc = stdevoflist(acc)
st_pre = stdevoflist(pre)
st_rec = stdevoflist(rec)
st_f1 = stdevoflist(f1)

print(st_acc, st_pre, st_rec, st_f1)
