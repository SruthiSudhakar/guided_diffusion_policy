import json
import pdb, sys

ogdict = json.load(open(f'{sys.argv[1]}/eval_log.json'))

aggs = {}
for k, v in ogdict.items():
    # pdb.set_trace()
    if 'train/sim_max_reward' in k:
        indx = k.split('_')[3]
        if indx not in aggs:
            aggs[indx]=[float(v)]
        elif len(aggs[indx])<int(sys.argv[2]):
            aggs[indx].append(float(v))
for k, v in aggs.items():
    aggs[k]=max(v)
# print(aggs)
print('max over samples', sum(aggs.values())/len(aggs.values()))


aggs = {}
for k, v in ogdict.items():
    # pdb.set_trace()
    if 'train/sim_max_reward' in k:
        indx = k.split('_')[3]
        if indx not in aggs:
            aggs[indx]=[float(v)]
        elif len(aggs[indx])<int(sys.argv[2]):
            aggs[indx].append(float(v))
for k, v in aggs.items():
    aggs[k]=sum(v)/len(v)
# print(aggs)
print('avg over samples', sum(aggs.values())/len(aggs.values()))