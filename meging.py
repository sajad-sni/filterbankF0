import pandas as pd

subs = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16']
Stimulus = '/MH/'
pathout = './data2/'

for i in subs:
    path1 = './data/' + i + Stimulus + '/Session1/' + i + '.csv'
    path2 = './data/' + i + Stimulus + '/Session2/' + i + '.csv'
    s1 = pd.read_csv(path1)
    s2 = pd.read_csv(path2)
    s1['Session'] = 1
    s2['Session'] = 2
    mixed_df = pd.concat([s1, s2]).to_csv(pathout+i+'.csv', index=False)
