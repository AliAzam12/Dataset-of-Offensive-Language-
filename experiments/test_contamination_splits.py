import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

df_canon = pd.read_csv('Dataset/canonical/full_data.csv')
df_rem = pd.read_csv('Dataset/audit/removed_variants.csv')
common_cols = ['clean_text', 'independence_group_id', 'language', 'label_id']
df_all = pd.concat([df_canon[common_cols], df_rem[common_cols]], ignore_index=True)

# Naive 80/20
train_naive, test_naive = train_test_split(df_all, test_size=0.20, random_state=42, stratify=df_all['label_id'])

# Group-aware 80/20 on the same 11,789 rows
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(df_all, df_all['label_id'], groups=df_all['independence_group_id']))
train_grp, test_grp = df_all.iloc[train_idx], df_all.iloc[test_idx]

train_naive_grps = set(train_naive['independence_group_id'])
train_grp_grps = set(train_grp['independence_group_id'])

naive_ov = test_naive['independence_group_id'].isin(train_naive_grps).mean() * 100
grp_ov = test_grp['independence_group_id'].isin(train_grp_grps).mean() * 100

print(f"Total instances: {len(df_all)}")
print(f"Condition A (Naive Split): Train={len(train_naive)}, Test={len(test_naive)}, Overlap={naive_ov:.2f}%")
print(f"Condition B (Group-Aware Split): Train={len(train_grp)}, Test={len(test_grp)}, Overlap={grp_ov:.2f}%\n")

for model_name, clf in [('LinearSVC (C=0.1, class_weight=None)', LinearSVC(C=0.1, random_state=42, max_iter=2000)),
                        ('Logistic Regression (C=1.0)', LogisticRegression(C=1.0, random_state=42, max_iter=2000)),
                        ('MultinomialNB (alpha=0.1)', MultinomialNB(alpha=0.1))]:
    # Evaluate with standard TF-IDF
    vec_n = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_tr_n = vec_n.fit_transform(train_naive['clean_text'])
    X_te_n = vec_n.transform(test_naive['clean_text'])
    clf.fit(X_tr_n, train_naive['label_id'])
    preds_n = clf.predict(X_te_n)
    f1_n = f1_score(test_naive['label_id'], preds_n, average='macro')
    
    vec_g = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_tr_g = vec_g.fit_transform(train_grp['clean_text'])
    X_te_g = vec_g.transform(test_grp['clean_text'])
    clf.fit(X_tr_g, train_grp['label_id'])
    preds_g = clf.predict(X_te_g)
    f1_g = f1_score(test_grp['label_id'], preds_g, average='macro')
    
    print(f"=== {model_name} ===")
    print(f"Overall: Naive F1 = {f1_n:.4f}, Group-Aware F1 = {f1_g:.4f}, Delta = {f1_n - f1_g:+.4f}")
    
    test_n_copy = test_naive.copy()
    test_n_copy['pred'] = preds_n
    test_g_copy = test_grp.copy()
    test_g_copy['pred'] = preds_g
    
    for lang in ['Roman Urdu', 'English', 'Urdu', 'Pashto']:
        sub_n = test_n_copy[test_n_copy['language'] == lang]
        sub_g = test_g_copy[test_g_copy['language'] == lang]
        sub_f1_n = f1_score(sub_n['label_id'], sub_n['pred'], average='macro')
        sub_f1_g = f1_score(sub_g['label_id'], sub_g['pred'], average='macro')
        ov_n = sub_n['independence_group_id'].isin(train_naive_grps).mean() * 100
        ov_g = sub_g['independence_group_id'].isin(train_grp_grps).mean() * 100
        print(f"  {lang:12s}: Naive F1 = {sub_f1_n:.4f} (ov={ov_n:.1f}%), Group F1 = {sub_f1_g:.4f} (ov={ov_g:.1f}%), Delta = {sub_f1_n - sub_f1_g:+.4f}")
    print()
