import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

# Load both datasets to get the full 11,789 rows
df_canon = pd.read_csv('Dataset/canonical/full_data.csv')
df_rem = pd.read_csv('Dataset/audit/removed_variants.csv')

common_cols = ['clean_text', 'independence_group_id', 'language', 'label_id']
df_all = pd.concat([df_canon[common_cols], df_rem[common_cols]], ignore_index=True)
n_total = len(df_all)
n_groups = df_all['independence_group_id'].nunique()
print(f'Total combined dataset: {n_total} rows, unique groups: {n_groups}')

# Condition A: Naive Random 80/20 Split
train_naive, test_naive = train_test_split(df_all, test_size=0.20, random_state=42, stratify=df_all['label_id'])
naive_train_groups = set(train_naive['independence_group_id'])
naive_test_overlap = test_naive['independence_group_id'].isin(naive_train_groups).mean() * 100
print(f'Condition A (Naive Split): Train={len(train_naive)}, Test={len(test_naive)}, Test Template Overlap={naive_test_overlap:.2f}%')

# Condition B: Group-Aware 80/20 Split on the same 11,789 rows
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(df_all, df_all['label_id'], groups=df_all['independence_group_id']))
train_group = df_all.iloc[train_idx]
test_group = df_all.iloc[test_idx]
group_train_groups = set(train_group['independence_group_id'])
group_test_overlap = test_group['independence_group_id'].isin(group_train_groups).mean() * 100
print(f'Condition B (Group-Aware Split): Train={len(train_group)}, Test={len(test_group)}, Test Template Overlap={group_test_overlap:.2f}%')

# Evaluation helper
def eval_split(train_df, test_df, split_name):
    # Using word + char n-grams as defined in paper
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_train = vec.fit_transform(train_df['clean_text'])
    X_test = vec.transform(test_df['clean_text'])
    y_train = train_df['label_id']
    y_test = test_df['label_id']
    
    clf = LinearSVC(C=0.1, random_state=42, max_iter=2000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    macro_f1 = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)
    print(f'\n[{split_name}] Overall Macro-F1: {macro_f1:.4f}, Accuracy: {acc:.4f}')
    
    results = {'Overall': {'f1': macro_f1, 'acc': acc, 'n': len(test_df), 'overlap': test_df['independence_group_id'].isin(set(train_df['independence_group_id'])).mean() * 100}}
    test_df_copy = test_df.copy()
    test_df_copy['pred'] = preds
    for lang in ['Roman Urdu', 'English', 'Urdu', 'Pashto']:
        sub = test_df_copy[test_df_copy['language'] == lang]
        sub_f1 = f1_score(sub['label_id'], sub['pred'], average='macro')
        sub_acc = accuracy_score(sub['label_id'], sub['pred'])
        overlap = sub['independence_group_id'].isin(set(train_df['independence_group_id'])).mean() * 100
        print(f'  {lang:12s} (N={len(sub):4d}): Macro-F1 = {sub_f1:.4f}, Acc = {sub_acc:.4f}, Overlap = {overlap:.1f}%')
        results[lang] = {'f1': sub_f1, 'acc': sub_acc, 'n': len(sub), 'overlap': overlap}
    return results

print('\n=== EVALUATING CONDITION A (NAIVE RANDOM SPLIT, N=11,789) ===')
res_naive = eval_split(train_naive, test_naive, 'Naive Random Split (N=11,789)')

print('\n=== EVALUATING CONDITION B (GROUP-AWARE SPLIT, N=11,789) ===')
res_group = eval_split(train_group, test_group, 'Group-Aware Split (N=11,789)')

print('\n=== DIRECT CONTROLLED COMPARISON (SAME N=11,789) ===')
print(f'Slice        Naive F1    Group-Aware F1    Delta F1 (Inflation)   Overlap Naive vs Group')
for k in ['Roman Urdu', 'English', 'Urdu', 'Pashto', 'Overall']:
    f1_n = res_naive[k]['f1']
    f1_g = res_group[k]['f1']
    d_f1 = f1_n - f1_g
    ov_naive = res_naive[k]['overlap']
    ov_group = res_group[k]['overlap']
    print(f"{k:12s} {f1_n:.4f}      {f1_g:.4f}            {d_f1:+.4f}               {ov_naive:.1f}% vs {ov_group:.1f}%")
