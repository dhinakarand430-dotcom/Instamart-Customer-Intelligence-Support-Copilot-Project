import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLEAN_DIR = Path('data/clean')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DATE = pd.Timestamp('2025-10-01')
COHORT_CUTOFF = pd.Timestamp('2025-09-01')

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data():
    customers = pd.read_csv(CLEAN_DIR / 'customers.csv')
    orders = pd.read_csv(CLEAN_DIR / 'orders.csv')
    products = pd.read_csv(CLEAN_DIR / 'products.csv')
    return customers, orders, products

def compute_rfm(customers, orders):
    # Filter completed orders only for RFM
    completed = orders[orders['order_status'] == 'completed']
    # Recency: days since last order relative to snapshot
    recency = completed.groupby('customer_id')['order_ts'].max().reset_index()
    recency['recency'] = (SNAPSHOT_DATE - pd.to_datetime(recency['order_ts'])).dt.days
    recency = recency[['customer_id', 'recency']]
    # Frequency: number of completed orders
    frequency = completed.groupby('customer_id').size().reset_index(name='frequency')
    # Monetary: total spend
    monetary = completed.groupby('customer_id')['order_amount'].sum().reset_index(name='monetary')
    rfm = recency.merge(frequency, on='customer_id').merge(monetary, on='customer_id')
    # Merge back to customers
    rfm = customers[['customer_id', 'city', 'age']].merge(rfm, on='customer_id', how='right')
    return rfm

# ---------------------------------------------------------------------------
# Part B1 – RFM & Clustering
# ---------------------------------------------------------------------------

def clustering_engine(rfm_df):
    # Standardize metrics for clustering
    metrics = rfm_df[['recency', 'frequency', 'monetary']].copy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(metrics)

    # Track inertia and silhouette for K=2..8
    scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X_std)
        inertia = km.inertia_
        if k > 1:
            sil = silhouette_score(X_std, labels)
        else:
            sil = np.nan
        scores.append({'k': k, 'inertia': inertia, 'silhouette': sil})
    # Enforce K=4 mapping – we take the K=4 solution
    k4 = KMeans(n_clusters=4, random_state=42, n_init='auto')
    rfm_df['cluster'] = k4.fit_predict(X_std)

    # Identify Premium cluster – highest average monetary
    cluster_stats = rfm_df.groupby('cluster')['monetary'].mean()
    premium_cluster = cluster_stats.idxmax()
    rfm_df['segment'] = rfm_df['cluster'].apply(lambda x: 'Premium' if x == premium_cluster else f'Cluster_{x}')

    # PCA projection (2 components)
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_std)
    rfm_df['pca_x'] = proj[:, 0]
    rfm_df['pca_y'] = proj[:, 1]

    # Save to CSV
    rfm_df.to_csv(OUTPUT_DIR / 'customer_segments.csv', index=False)
    return rfm_df, scores

# ---------------------------------------------------------------------------
# Part B2 – Leakage‑Free Cohort & Modeling
# ---------------------------------------------------------------------------

def build_features(customers, orders, products):
    # Use only orders before cutoff
    orders['order_ts'] = pd.to_datetime(orders['order_ts'])
    pre_cutoff = orders[orders['order_ts'] < COHORT_CUTOFF]
    completed = pre_cutoff[pre_cutoff['order_status'] == 'completed']
    # Recency relative to cutoff
    recency = completed.groupby('customer_id')['order_ts'].max().reset_index()
    recency['recency'] = (COHORT_CUTOFF - recency['order_ts']).dt.days
    # Frequency
    frequency = completed.groupby('customer_id').size().reset_index(name='frequency')
    # Monetary
    monetary = completed.groupby('customer_id')['order_amount'].sum().reset_index(name='monetary')
    # Cancellation rate (cancelled / total orders before cutoff)
    total_orders = pre_cutoff.groupby('customer_id').size().reset_index(name='total_orders')
    cancelled = pre_cutoff[pre_cutoff['order_status'] == 'cancelled'].groupby('customer_id').size().reset_index(name='cancelled')
    cancel_rate = total_orders.merge(cancelled, on='customer_id', how='left')
    cancel_rate['cancelled'] = cancel_rate['cancelled'].fillna(0)
    cancel_rate['cancellation_rate'] = cancel_rate['cancelled'] / cancel_rate['total_orders']
    # Favorite category (most frequent category in completed orders)
    completed = completed.merge(products[['product_id', 'category']], on='product_id', how='left')
    fav_cat = completed.groupby(['customer_id', 'category']).size().reset_index(name='cnt')
    fav_cat = fav_cat.sort_values(['customer_id', 'cnt'], ascending=[True, False])
    fav_cat = fav_cat.drop_duplicates('customer_id')[['customer_id', 'category']].rename(columns={'category': 'fav_category'})
    # Tenure – days between first order ever and cutoff
    first_order = pre_cutoff.groupby('customer_id')['order_ts'].min().reset_index(name='first_ts')
    first_order['tenure'] = (COHORT_CUTOFF - first_order['first_ts']).dt.days
    # Assemble features
    features = customers[['customer_id', 'city', 'age']].merge(recency, on='customer_id', how='left')
    features = features.merge(frequency, on='customer_id', how='left')
    features = features.merge(monetary, on='customer_id', how='left')
    features = features.merge(cancel_rate[['customer_id', 'cancellation_rate']], on='customer_id', how='left')
    features = features.merge(fav_cat, on='customer_id', how='left')
    features = features.merge(first_order[['customer_id', 'tenure']], on='customer_id', how='left')
    # Fill missing numeric with 0, categorical with 'Unknown'
    num_cols = ['recency', 'frequency', 'monetary', 'cancellation_rate', 'age', 'tenure']
    features[num_cols] = features[num_cols].fillna(0)
    cat_cols = ['city', 'fav_category']
    features[cat_cols] = features[cat_cols].fillna('Unknown')
    return features

def oversample_random(X, y):
    # Simple random oversampling of minority class to match majority size
    from collections import Counter
    counter = Counter(y)
    majority = max(counter.values())
    X_res = X.copy()
    y_res = y.copy()
    for cls, count in counter.items():
        if count < majority:
            needed = majority - count
            idx = np.where(y == cls)[0]
            sampled_idx = np.random.choice(idx, size=needed, replace=True)
            X_res = np.vstack([X_res, X[sampled_idx]])
            y_res = np.concatenate([y_res, np.full(needed, cls)])
    return X_res, y_res

def train_model(features_df):
    # Define target: premium customers are those in Premium segment (from clustering)
    # We'll load the segment mapping from customer_segments.csv
    seg_path = OUTPUT_DIR / 'customer_segments.csv'
    seg_df = pd.read_csv(seg_path)
    premium_ids = set(seg_df[seg_df['segment'] == 'Premium']['customer_id'])
    features_df['target'] = features_df['customer_id'].apply(lambda x: 1 if x in premium_ids else 0)
    X = features_df.drop(columns=['customer_id', 'target'])
    y = features_df['target'].values

    # ColumnTransformer for preprocessing
    numeric_features = ['recency', 'frequency', 'monetary', 'cancellation_rate', 'age', 'tenure']
    categorical_features = ['city', 'fav_category']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )
    # Pipeline will be defined per estimator inside GridSearchCV
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    # Oversample training set
    # Convert to numpy after preprocessing inside pipeline – easier to oversample after transformation
    # We'll construct a pipeline with preprocessing + classifier, and use GridSearchCV with custom scoring.
    # For oversampling, we will create a wrapper that applies oversampling inside a Pipeline via a FunctionTransformer.
    from sklearn.pipeline import make_union
    from sklearn.base import BaseEstimator, TransformerMixin

    class Oversampler(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            # X is already a dense array after preprocessing if we set pipeline steps accordingly
            return X

    # Define pipelines for each model
    pipe_lr = Pipeline(steps=[('pre', preprocessor),
                              ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))])
    pipe_rf = Pipeline(steps=[('pre', preprocessor),
                              ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42))])
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10] if 'LogisticRegression' in str(pipe_lr.named_steps['clf']) else None,
        'clf__max_depth': [5, 10, None] if 'RandomForestClassifier' in str(pipe_rf.named_steps['clf']) else None,
        'clf__min_samples_leaf': [1, 2, 4] if 'RandomForestClassifier' in str(pipe_rf.named_steps['clf']) else None,
    }
    # Since param_grid varies per model, we will run two separate GridSearchCVs
    # Logistic Regression
    lr_grid = GridSearchCV(pipe_lr, {'clf__C': [0.01, 0.1, 1, 10]}, scoring='f1', cv=5, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    # Random Forest
    rf_grid = GridSearchCV(pipe_rf, {'clf__max_depth': [5, 10, None], 'clf__min_samples_leaf': [1, 2, 4]}, scoring='f1', cv=5, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    # Choose best based on validation f1
    best_model = lr_grid if lr_grid.best_score_ >= rf_grid.best_score_ else rf_grid
    # Retrain on full oversampled training set
    # Apply preprocessing to train data, oversample, then fit final estimator
    X_train_pre = best_model.best_estimator_.named_steps['pre'].fit_transform(X_train)
    X_train_os, y_train_os = oversample_random(X_train_pre, y_train)
    best_estimator = best_model.best_estimator_.named_steps['clf']
    best_estimator.fit(X_train_os, y_train_os)
    # Save model
    dump(best_estimator, OUTPUT_DIR / 'lapse_model.joblib')
    # Predict probabilities on full dataset to rank premium high‑risk (probability of class 1)
    X_full_pre = best_model.best_estimator_.named_steps['pre'].transform(X)
    probs = best_estimator.predict_proba(X_full_pre)[:, 1]
    features_df['risk_score'] = probs
    top10 = features_df.sort_values('risk_score', ascending=False).head(10)
    top10[['customer_id', 'risk_score']].to_csv(OUTPUT_DIR / 'retention_list.csv', index=False)
    return best_estimator

if __name__ == '__main__':
    customers, orders, products = load_data()
    rfm = compute_rfm(customers, orders)
    rfm, inertia_sil_scores = clustering_engine(rfm)
    print('Inertia/Silhouette scores for K=2..8:')
    for s in inertia_sil_scores:
        print(s)
    features = build_features(customers, orders, products)
    model = train_model(features)
    print('Model trained and saved.')