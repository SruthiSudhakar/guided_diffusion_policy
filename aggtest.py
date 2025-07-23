import json
import pdb, sys

ogdict = json.load(open(f'{sys.argv[1]}/eval_log.json'))

aggs = {}
for k, v in ogdict.items():
    if 'train/sim_max_reward' in k:
        indx = k.split('_')[3]
        if indx not in aggs:
            aggs[indx]=[float(v)]
        elif len(aggs[indx])<int(sys.argv[2]):
            aggs[indx].append(float(v))
for k, v in aggs.items():
    if 1<=sum(v)<=9:
        print(k)
    aggs[k]=max(v)
print(f'TRAIN max over {len(v)} samples', sum(aggs.values())/len(aggs.values()))


aggs = {}
for k, v in ogdict.items():
    if 'train/sim_max_reward' in k:
        indx = k.split('_')[3]
        if indx not in aggs:
            aggs[indx]=[float(v)]
        elif len(aggs[indx])<int(sys.argv[2]):
            aggs[indx].append(float(v))
for k, v in aggs.items():
    aggs[k]=sum(v)/len(v)
print(f'TRAIN avg over {len(v)} samples', sum(aggs.values())/len(aggs.values()))

aggs = {}
for k, v in ogdict.items():
    if 'train/sim_max_reward' in k:
        indx = k.split('_')[3]
        if indx not in aggs:
            aggs[indx]=[float(v)]
        elif len(aggs[indx])<int(sys.argv[2]):
            aggs[indx].append(float(v))

for k, v in aggs.items():
    aggs[k]=sum(v)/len(v)

aggs = {k: aggs[k] for k in sorted(aggs.keys(), key=lambda x: int(x))}
print(f'TRAIN mean over first {sys.argv[3]} rollouts', sum(list(aggs.values())[:int(sys.argv[3])])/len(list(aggs.values())[:int(sys.argv[3])]))



aggs = {}
for k, v in ogdict.items():
    if 'test/sim_max_reward' in k:
        indx = k.split('_')[3]
        if indx not in aggs:
            aggs[indx]=[float(v)]
        elif len(aggs[indx])<int(sys.argv[2]):
            aggs[indx].append(float(v))
for k, v in aggs.items():
    aggs[k]=max(v)
print(f'TEST max over {len(v)} samples', sum(aggs.values())/len(aggs.values()))


aggs = {}
for k, v in ogdict.items():
    if 'test/sim_max_reward' in k:
        indx = k.split('_')[3]
        if indx not in aggs:
            aggs[indx]=[float(v)]
        elif len(aggs[indx])<int(sys.argv[2]):
            aggs[indx].append(float(v))
for k, v in aggs.items():
    aggs[k]=sum(v)/len(v)
print(f'TEST avg over {len(v)} samples', sum(aggs.values())/len(aggs.values()))

aggs = {}
for k, v in ogdict.items():
    if 'test/sim_max_reward' in k:
        indx = k.split('_')[3]
        if indx not in aggs:
            aggs[indx]=[float(v)]
        elif len(aggs[indx])<int(sys.argv[2]):
            aggs[indx].append(float(v))

for k, v in aggs.items():
    aggs[k]=sum(v)/len(v)

aggs = {k: aggs[k] for k in sorted(aggs.keys(), key=lambda x: int(x))}
print(f'TEST mean over first {sys.argv[3]} rollouts', sum(list(aggs.values())[:int(sys.argv[3])])/len(list(aggs.values())[:int(sys.argv[3])]))