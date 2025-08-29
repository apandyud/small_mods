import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import cga_utils
import math
from typing import Iterable, List, Optional, Tuple
import hdbscan
from sklearn.feature_extraction import DictVectorizer

def err_bucket(rel_err):
    if rel_err < 1e-6: 
        return 0
    elif rel_err < 0.01:
        return 1
    elif rel_err < 0.1:
        return 2
    elif rel_err < 1:
        return 3
    else:
        return 4

def value_bucket(ans, pred):
    if ans == pred:
        return 0
    if ans == -1*pred:
        return 1
    if math.isnan(pred):
        return 2
    else: 
        return 3

def prepare_matrix(errors, features):
    EPS = 1e-9
    
    errors["rel_err"] = errors.apply(lambda row: (row["pred"] - row["answer"]) / max(row["answer"], EPS), axis=1  )
    errors["abs_err"] = errors.apply(lambda row: row["pred"] - row["answer"], axis=1  )
    errors["magnitude_bucket"] = errors.apply(lambda row: int(math.floor(math.log10(abs(row["answer"])+EPS))) if row["answer"] != 0 else -1, axis=1  )
    errors["rel_error_bucket"] = errors["rel_err"].apply(err_bucket)
    errors["ratio"] = errors.apply(lambda row: (row["pred"] / (row["answer"]+EPS)) if row["answer"] != 0 else np.inf,  axis=1 )
    errors["x100_flag"] = errors["ratio"].apply(lambda ratio: int(0.95 < ratio/100 < 1.05 or 0.95 < ratio*100 < 1.05))
    errors["has_error_text"] = errors["error_text"].apply(lambda txt: txt != None and txt != '')
    errors["ratio_is_inf"] = np.isinf(errors["ratio"]).astype(int)
    errors["ratio"] = np.where(np.isinf(errors["ratio"]), np.nan, errors["ratio"])
    
    errors["scale_mismatch"]= errors["scale"] != errors["pred_scale"]
    errors["value_mismatch"]= errors["answer"] != errors["pred"]
    errors["value_nan"]= errors["pred"].isna()
    errors["value_match"]= errors.apply(lambda row: value_bucket(row["answer"], row["pred"]) ,  axis=1 )

    # 2) Erősen ferde oszlopok log1p-vel és/vagy winsorize/clip
    for col in ["abs_err", "rel_err", "ratio"]:
        if col in errors:
            # negatív is lehet -> signed log1p
            errors[col] = np.sign(errors[col]) * np.log1p(np.abs(errors[col]))

    # 3) NaN-ek kezelése (pl. median impute)
    num_cols = ["rel_err","abs_err","ratio"]
    for col in num_cols:
        if col in errors:
            med = errors[col].median()
            errors[col] = errors[col].fillna(med)

    # 4) Skálázás a folytonosokra (RobustScaler a kilógók ellen)
    from sklearn.preprocessing import RobustScaler
    cont = errors[num_cols].values
    scaler = RobustScaler().fit(cont)
    errors[num_cols] = scaler.transform(cont)

    errors["scale"] = errors["scale"].fillna("")
    errors["pred_scale"] = errors["pred_scale"].fillna("")
    errors["code_calc_pattern"]= errors["code_calc_pattern"].fillna("")

    #vects = errors.drop(["ts", "qid", "pred_scale", "question", "derivation", "calc_pattern", "pred", "answer", "scale", "value_match", "error_text",	"value_list", "code",	"selected_values","needed_values","exact_match",	"error_code"], axis=1)
    #vects = errors[["calc_pattern", "error_code",  "pred_ast", "selection_success", "sign_error", "is_parenth_in_table", "has_code_abs",
    # "scale", "pred_scale",  "rel_error_bucket", "magnitude_bucket", "x100_flag",  "has_error_text"]]
    #vects = errors[["calc_pattern", "code_calc_pattern", "scale", "pred_scale"]]
    #vects = errors[["calc_pattern", "code_calc_pattern", "scale", "pred_scale","x100_flag", "sign_error", 'error_code']]
    vects = errors[features]
    X_dict = vects.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X_flags = dv.fit_transform(X_dict)   
    return X_flags

def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 8, min_samples: int = 2) -> Tuple[np.ndarray, Optional[float]]:
    if hdbscan is None:
        print("[warn] hdbscan not installed; falling back to agglomerative.")
        return cluster_agglomerative(X, min_cluster_size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="manhattan")
    labels = clusterer.fit_predict(X)
    print('pers: ', clusterer.cluster_persistence_)
    return labels, clusterer.cluster_persistence_, clusterer.probabilities_

from typing import Iterable, List, Optional, Tuple

def try_hdbscan(X, mcs_list=(2,3,4, 5,6, 10,15), ms_list=(1,2,3, 4, 5)):
    from collections import defaultdict
    out = []
    for mcs in mcs_list:
        for ms in ms_list:
            cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="manhattan")
            labels = cl.fit_predict(X)
            n_clusters = len(set(labels) - {-1})
            noise_ratio = (labels == -1).mean()
            out.append((mcs, ms, n_clusters, noise_ratio))        
    return pd.DataFrame(out).sort_values(by=3)