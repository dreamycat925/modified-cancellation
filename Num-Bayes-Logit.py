# divergence対策としてお手つきだけはずしてみる

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# ✅ 健常群 vs 患者群で分類するためのデータ抽出
Z = df_com[df_com['group'].isin(['hs', 'ci'])].copy()

# ✅ 説明変数（カテゴリ変数をダミー化）
X = Z[['検査時の年齢', '性別', '教育歴', 
       'my_cancellation_数字_所要時間',
       'my_cancellation_数字_正答数']]

X = pd.get_dummies(X, columns=['性別'], drop_first=True, dtype=int)

# ✅ 目的変数 (患者群=1, 健常群=0)
y = (Z['group'] == 'ci').astype(int).values  

# ✅ 数値変数の標準化
num_cols = ['検査時の年齢', '教育歴', 
            'my_cancellation_数字_所要時間', 
            'my_cancellation_数字_正答数']

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ✅ PyMC ベイズロジスティック回帰
with pm.Model() as bayesian_logit_model:
    # 事前分布設定
    intercept = pm.Normal("intercept", mu=0, sigma=2)
    β_age = pm.Normal("β_age", mu=0, sigma=1)
    β_edu = pm.Normal("β_edu", mu=0, sigma=1)
    β_gender = pm.Normal("β_gender", mu=0, sigma=1)
    β_time = pm.Normal("β_time", mu=0, sigma=1)
    β_correct = pm.Normal("β_correct", mu=0, sigma=1)

    # 線形結合
    μ = (
        intercept
        + β_age * X["検査時の年齢"].values
        + β_edu * X["教育歴"].values
        + β_gender * X["性別_男"].values
        + β_time * X['my_cancellation_数字_所要時間'].values
        + β_correct * X['my_cancellation_数字_正答数'].values
    )

    # シグモイド関数で確率を得る
    p = pm.math.sigmoid(μ)

    # 尤度関数（ベルヌーイ分布）
    likelihood = pm.Bernoulli("y", p=p, observed=y)

    # MCMCサンプリング
    trace_bayes_logit = pm.sample(
        4000, tune=2000, chains=4, target_accept=0.995, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

# ✅ 事後分布の要約
summary = az.summary(trace_bayes_logit, stat_funcs={"median": np.median}, hdi_prob=0.95, round_to=4)
print(summary)

# ✅ オッズ比の計算
odds_ratios = np.exp(summary['mean'])
print("\nオッズ比:\n", odds_ratios)

odds_ratios_lower = np.exp(summary['hdi_2.5%'])
print("\nオッズ比下限:\n", odds_ratios_lower)

odds_ratios_upper = np.exp(summary['hdi_97.5%'])
print("\nオッズ比上限:\n", odds_ratios_upper)

# ✅ 事後確率の計算関数
def compute_posterior_probabilities(trace, param):
    samples = trace.posterior[param].values.flatten()
    prob_positive = (samples > 0).mean()
    prob_negative = (samples < 0).mean()
    return prob_positive, prob_negative

print("\n事後確率:")
for param in ["β_age", "β_edu", "β_gender", "β_time", "β_correct"]:
    p_pos, p_neg = compute_posterior_probabilities(trace_bayes_logit, param)
    print(f"{param}: P(β > 0) = {p_pos:.3f}, P(β < 0) = {p_neg:.3f}")

# ✅ 予測確率を取得 (事後予測分布から)
with bayesian_logit_model:
    posterior_pred = pm.sample_posterior_predictive(
        trace_bayes_logit, var_names=["y"], random_seed=42
    )

# 予測確率を算出
pred_prob = posterior_pred.posterior_predictive["y"].mean(dim=["chain", "draw"]).values

# ✅ ROC曲線の作成
fpr, tpr, thresholds = roc_curve(y, pred_prob)
roc_auc = auc(fpr, tpr)
print(f'\nAUC: {roc_auc:.3f}')

# ROC曲線のプロット
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=3)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('1 - Specificity', fontsize=16, fontweight='bold')
plt.ylabel('Sensitivity', fontsize=16, fontweight='bold')
plt.title('ROC Curve', fontsize=18, fontweight='bold')
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14, width=2)

# 目盛りのフォントサイズと太さを調整
plt.tick_params(axis='both', which='major', labelsize=22, width=2)

# 目盛りラベルを太字にする
for label in plt.gca().get_xticklabels():
    label.set_fontweight('bold')
for label in plt.gca().get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()

# トレースプロットと事後分布
az.plot_trace(trace_bayes_logit, figsize=(20, 12), combined=False, compact=False)
plt.tight_layout()

az.plot_posterior(trace_bayes_logit, hdi_prob=0.95)
plt.tight_layout()