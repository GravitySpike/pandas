import pandas as pd
df1=pd.read_csv("2019_kbo_for_kaggle_v2.csv")
isyear=df1['year']>=2015
isyear2=df1['year']<=2018
df2=df1[isyear & isyear2]
df2.sort_values(ascending=False, by='H').head(10)
df2.sort_values(ascending=False, by='avg').head(10)
df2.sort_values(ascending=False, by='HR').head(10)
df2.sort_values(ascending=False, by='OBP').head(10)

is2018=df1['year']==2018
isc=df1['cp']=='포수'
is1b=df1['cp']=='1루수'
is2b=df1['cp']=='2루수'
is3b=df1['cp']=='3루수'
isss=df1['cp']=='유격수'
islf=df1['cp']=='좌익수'
isrf=df1['cp']=='우익수'
iscf=df1['cp']=='중견수'

c=df1[is2018 & isc]
oneb=df1[is2018 & is1b]
twob=df1[is2018 & is2b]
threeb=df1[is2018 & is3b]
ss=df1[is2018 & isss]
lf=df1[is2018 & islf]
cf=df1[is2018 & iscf]
rf=df1[is2018 & isrf]

c.sort_values(ascending=False, by='war').head(1)
oneb.sort_values(ascending=False, by='war').head(1)
twob.sort_values(ascending=False, by='war').head(1)
threeb.sort_values(ascending=False, by='war').head(1)
ss.sort_values(ascending=False, by='war').head(1)
lf.sort_values(ascending=False, by='war').head(1)
cf.sort_values(ascending=False, by='war').head(1)
rf.sort_values(ascending=False, by='war').head(1)

df2=df1[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]
co_matrix=df2.corr()
print(co_matrix)

