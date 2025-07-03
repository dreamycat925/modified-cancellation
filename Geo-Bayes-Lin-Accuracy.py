# ベイズ線形回帰　幾何学図形　正答数を目的変数
# お手つきなし

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 健常群のみ抽出
df_hs = df_com[df_com['group'] == 'hs'].copy()

# 説明変数を指定（年齢、性別、教育歴、所要時間、正答数、お手つき数）
X = df_hs[['検査時の年齢', '性別', '教育歴',
           'my_cancellation_幾何学図形_所要時間',
           'my_cancellation_幾何学図形_正答数']].copy()

# カテゴリ変数（性別）のエンコーディング
X = pd.get_dummies(X, columns=['性別'], drop_first=True, dtype=int)

# 標準化
scaler = StandardScaler()
num_cols = ['検査時の年齢', '教育歴',
            'my_cancellation_幾何学図形_所要時間',
            'my_cancellation_幾何学図形_正答数']

X[num_cols] = scaler.fit_transform(X[num_cols])

# ベイズ線形回帰モデル
with pm.Model() as attention_model:
    beta_0 = pm.Normal("beta_0", mu=0, sigma=1)
    beta_age = pm.Normal("beta_age", mu=0, sigma=1)
    beta_edu = pm.Normal("beta_edu", mu=0, sigma=1)
    beta_gender = pm.Normal("beta_gender", mu=0, sigma=1)

    # 回帰式（所要時間を中心的な指標として総合スコアを作成）
    mu = (
        beta_0
        + beta_age * X["検査時の年齢"]
        + beta_edu * X["教育歴"]
        + beta_gender * X["性別_男"]
    )

    sigma = pm.HalfNormal("sigma", sigma=1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=X["my_cancellation_幾何学図形_正答数"])

    # MCMCサンプリング
    trace_attention = pm.sample(2000, tune=1000, chains=4,
                                return_inferencedata=True,
                                idata_kwargs={"log_likelihood": True})

# 事後分布の要約表示
summary_attention = az.summary(trace_attention, stat_funcs={"median": np.median}, hdi_prob=0.95)
print(summary_attention)

# 事後分布の可視化
az.plot_trace(trace_attention, figsize=(20, 12), combined=False, compact=False)
plt.tight_layout()
az.plot_posterior(trace_attention, hdi_prob=0.95)
plt.show()

# 事後確率の計算関数
def compute_posterior_probabilities(trace, param):
    samples = trace.posterior[param].values.flatten()
    prob_positive = (samples > 0).mean()
    prob_negative = (samples < 0).mean()
    return prob_positive, prob_negative

# 各パラメータの事後確率を表示
print("\n事後確率:")
for param in ["beta_age", "beta_edu", "beta_gender"]:
    p_pos, p_neg = compute_posterior_probabilities(trace_attention, param)
    print(f"{param}: P(β > 0) = {p_pos:.3f}, P(β < 0) = {p_neg:.3f}")