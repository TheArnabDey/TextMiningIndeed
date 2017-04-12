import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

df1 = pd.DataFrame(pd.read_table('train.tsv'))
df1.iloc[:,1] = df1.iloc[:,1].str.replace('.',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace(',',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('\'',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace(':',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace(';',' ')
#df.iloc[:,0] = df.iloc[:,0].str.replace('?',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('/',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('(',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('+',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('_',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('-',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('!',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('@',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('#',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('$',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('!',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('\*',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('%',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('&',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('|',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('"',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('<',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('>',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('\?',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace('~',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.replace(')',' ')
df1.iloc[:,1] = df1.iloc[:,1].str.lower()

df1['part'] = "m"
df1['full'] = "m"
df1['hourly'] = "m"
df1['salary'] = "m"
df1['associate'] = "m"
df1['bs'] = "m"
df1['ms'] = "m"
df1['license'] = "m"
df1['1year'] = "m"
df1['24years'] = "m"
df1['5years'] = "m"
df1['super'] = "m"

for x in df1.index:
    if str(df1.iloc[x,0]).find("part-time-job") != -1:
        df1.iloc[x, 2] = -1
    else:
        df1.iloc[x, 2] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("full-time-job") != -1:
        df1.iloc[x, 3] = 1
    else:
        df1.iloc[x, 3] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("hourly-wage") != -1:
        df1.iloc[x, 4] = -1
    else:
        df1.iloc[x, 4] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("salary") != -1:
        df1.iloc[x, 5] = 1
    else:
        df1.iloc[x, 5] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("associate-needed") != -1:
        df1.iloc[x, 6] = 1
    else:
        df1.iloc[x, 6] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("bs-degree-needed") != -1:
        df1.iloc[x, 7] = -1
    else:
        df1.iloc[x, 7] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("ms-or-phd-needed") != -1:
        df1.iloc[x, 8] = 1
    else:
        df1.iloc[x, 8] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("licence-needed") != -1:
        df1.iloc[x, 9] = 1
    else:
        df1.iloc[x, 9] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("1-year-experience-needed") != -1:
        df1.iloc[x, 10] = 1
    else:
        df1.iloc[x, 10] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("2-4-years-experience-needed") != -1:
        df1.iloc[x, 11] = 2
    else:
        df1.iloc[x, 11] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("5-plus-years-experience-needed") != -1:
        df1.iloc[x, 12] = 3
    else:
        df1.iloc[x, 12] = 0

for x in df1.index:
    if str(df1.iloc[x,0]).find("supervising-job") != -1:
        df1.iloc[x, 13] = 1
    else:
        df1.iloc[x, 13] = 0

print "step 1"

df = pd.DataFrame(pd.read_table('train.tsv'))
df.iloc[:,1] = df.iloc[:,1].str.replace('.',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace(',',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('\'',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace(':',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace(';',' ')
#df.iloc[:,0] = df.iloc[:,0].str.replace('?',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('/',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('(',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('+',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('_',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('-',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('!',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('@',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('#',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('$',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('!',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('\*',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('%',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('&',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('|',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('"',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('<',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('>',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('\?',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace('~',' ')
df.iloc[:,1] = df.iloc[:,1].str.replace(')',' ')
df.iloc[:,1] = df.iloc[:,1].str.lower()

df['part'] = "m"
df['full'] = "m"
df['hourly'] = "m"
df['salary'] = "m"
df['associate'] = "m"
df['bs'] = "m"
df['ms'] = "m"
df['license'] = "m"
df['1year'] = "m"
df['24years'] = "m"
df['5years'] = "m"
df['super'] = "m"

for x in df.index:
    if str(df.iloc[x,0]).find("part-time-job") != -1:
        df.iloc[x, 2] = -1
        df.iloc[x, 3] = -1
    else:
        df.iloc[x, 2] = 0
        df.iloc[x, 3] = 0

for x in df.index:
    if str(df.iloc[x,0]).find("full-time-job") != -1:
        df.iloc[x, 2] = 1
        df.iloc[x, 3] = 1

for x in df.index:
    if str(df.iloc[x,0]).find("hourly-wage") != -1:
        df.iloc[x, 4] = -1
        df.iloc[x, 5] = -1
    else:
        df.iloc[x, 4] = 0
        df.iloc[x, 5] = 0

for x in df.index:
    if str(df.iloc[x,0]).find("salary") != -1:
        df.iloc[x, 4] = 1
        df.iloc[x, 5] = 1

for x in df.index:
    if str(df.iloc[x,0]).find("associate-needed") != -1:
        df.iloc[x, 6] = 1
    else:
        df.iloc[x, 6] = 0

for x in df.index:
    if str(df.iloc[x,0]).find("bs-degree-needed") != -1:
        df.iloc[x, 7] = -1
        df.iloc[x, 8] = -1
    else:
        df.iloc[x, 7] = 0
        df.iloc[x, 8] = 0

for x in df.index:
    if str(df.iloc[x,0]).find("ms-or-phd-needed") != -1:
        df.iloc[x, 7] = 1
        df.iloc[x, 8] = 1

for x in df.index:
    if str(df.iloc[x,0]).find("licence-needed") != -1:
        df.iloc[x, 9] = 1
    else:
        df.iloc[x, 9] = 0

for x in df.index:
    if str(df.iloc[x,0]).find("1-year-experience-needed") != -1:
        df.iloc[x, 10] = 1
        df.iloc[x, 11] = 1
        df.iloc[x, 12] = 1
    else:
        df.iloc[x, 10] = 0
        df.iloc[x, 11] = 0
        df.iloc[x, 12] = 0

for x in df.index:
    if str(df.iloc[x,0]).find("2-4-years-experience-needed") != -1:
        df.iloc[x, 10] = 2
        df.iloc[x, 11] = 2
        df.iloc[x, 12] = 2

for x in df.index:
    if str(df.iloc[x,0]).find("5-plus-years-experience-needed") != -1:
        df.iloc[x, 10] = 3
        df.iloc[x, 11] = 3
        df.iloc[x, 12] = 3

for x in df.index:
    if str(df.iloc[x,0]).find("supervising-job") != -1:
        df.iloc[x, 13] = 1
    else:
        df.iloc[x, 13] = 0

print "step 2"

dtf = pd.DataFrame(pd.read_table('test.tsv'))
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('.',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace(',',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('\'',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace(':',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace(';',' ')
#df.iloc[:,0] = df.iloc[:,0].str.replace('?',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('/',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('(',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('+',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('_',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('-',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('!',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('@',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('#',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('$',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('!',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('\*',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('%',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('&',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('|',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('"',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('<',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('>',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('\?',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace('~',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.replace(')',' ')
dtf.iloc[:,0] = dtf.iloc[:,0].str.lower()

dtf['part'] = 10
dtf['full'] = 10
dtf['hourly'] = 10
dtf['salary'] = 10
dtf['associate'] = 10
dtf['bs'] = 10
dtf['ms'] = 10
dtf['license'] = 10
dtf['1year'] = 10
dtf['24years'] = 10
dtf['5years'] = 10
dtf['super'] = 10

print "step 3"

features_train = df.iloc[:, 1]
features_train1 = df1.iloc[:, 1]
features_test = dtf.iloc[:,0]
vectorizer = TfidfVectorizer(sublinear_tf=True)
features_train_transformed = vectorizer.fit_transform(features_train)
features_train_transformed1 = vectorizer.fit_transform(features_train1)
features_test_transformed  = vectorizer.transform(features_test)
selector = SelectPercentile(f_classif, percentile=20)
selector1 = SelectPercentile(f_classif, percentile=20)
clf = GradientBoostingClassifier(max_depth=10, min_samples_split= 500, n_estimators=200)
clf1 = GradientBoostingClassifier(max_depth=10, min_samples_split= 500, n_estimators=200)

print "step 4"

for i in range(2,14):
    labels_train = df.iloc[:,i]
    labels_train1 = df1.iloc[:,i]
    selector.fit(features_train_transformed, labels_train)
    selector1.fit(features_train_transformed1, labels_train1)
    features_train_final = selector.transform(features_train_transformed).toarray()
    features_train_final1 = selector1.transform(features_train_transformed1).toarray()
    clf.fit(features_train_final, np.asarray(labels_train, dtype="|S6"))
    clf1.fit(features_train_final1, np.asarray(labels_train1, dtype="|S6"))
    features_test_final = selector.transform(features_test_transformed).toarray()
    features_test_final1 = selector1.transform(features_test_transformed).toarray()
    for x in dtf.index:
        pred1 = clf.predict_proba(features_test_final[x])
        pred2 = clf1.predict_proba(features_test_final1[x])
        if max(pred2[0]) > max(pred1[0]):
            dtf.iloc[x,i-1] = clf1.predict(features_test_final1[x])
        else:
            dtf.iloc[x,i-1] = clf.predict(features_test_final[x])
    #print i

print "step 5"

dtf['tags'] = " "
a = 0

for x in dtf.index:
    k = 0
    if dtf.iloc[x,1] == '-1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " part-time-job"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "part-time-job"
            k = 1
            a = a + 1
    if dtf.iloc[x,2] == '1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " full-time-job"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "full-time-job"
            k = 1
            a = a + 1
    if dtf.iloc[x, 3] == '-1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " hourly-wage"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "hourly-wage"
            k = 1
            a = a + 1
    if dtf.iloc[x, 4] == '1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " salary"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "salary"
            k = 1
            a = a + 1
    if dtf.iloc[x, 5] == '1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " associate-needed"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "associate-needed"
            k = 1
            a = a + 1
    if dtf.iloc[x, 7] == '1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " ms-or-phd-needed"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "ms-or-phd-needed"
            k = 1
            a = a + 1
    if dtf.iloc[x, 6] == '-1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " bs-degree-needed"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "bs-degree-needed"
            k = 1
            a = a + 1
    if dtf.iloc[x, 8] == '1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " licence-needed"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "licence-needed"
            k = 1
            a = a + 1
    if dtf.iloc[x, 9] == '1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " 1-year-experience-needed"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "1-year-experience-needed"
            k = 1
            a = a + 1
    if dtf.iloc[x, 11] == '3':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " 5-plus-years-experience-needed"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "5-plus-years-experience-needed"
            k = 1
            a = a + 1
    if dtf.iloc[x, 10] == '2':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " 2-4-years-experience-needed"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "2-4-years-experience-needed"
            k = 1
            a = a + 1
    if dtf.iloc[x, 12] == '1':
        if k == 1:
            dtf.iloc[x, 13] = dtf.iloc[x, 13] + " supervising-job"
            a = a + 1
        else:
            dtf.iloc[x, 13] = "supervising-job"
            k = 1
            a = a + 1

print a
print dtf["tags"]

dtf.to_csv("tags.tsv", columns= ["tags"], sep="\t", index= False)
