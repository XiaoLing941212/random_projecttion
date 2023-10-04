import random
import numpy as np
import math
import os
import pandas as pd
import time
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

def cluster(candidates, enough, res):
    if len(candidates) < enough:
        res.append(candidates)
        return res

    east, west, east_items, west_items = split(candidates)
    res = cluster(east_items, enough, res)
    res = cluster(west_items, enough, res)

    return res


def split(candidates):
    pivot = random.choice(candidates)
    east_pivot = find_farest(pivot, candidates)
    west_pivot = find_farest(east_pivot, candidates)
    c = cal_distance(east_pivot, west_pivot)

    if c == 0:
        east_items = candidates[:len(candidates)//2]
        west_items = candidates[len(candidates)//2:]
        return east_pivot, west_pivot, east_items, west_items

    all_distance = []
    for candidate in candidates:
        a = cal_distance(candidate, west_pivot)
        b = cal_distance(candidate, east_pivot)
        d = (a ** 2 + c ** 2 - b ** 2) / (2 * c)
        all_distance.append((d, candidate))

    all_distance.sort(key=lambda x: x[0])
    sorted_candidates = [item[1] for item in all_distance]
    east_items = sorted_candidates[:len(sorted_candidates)//2]
    west_items = sorted_candidates[len(sorted_candidates)//2:]

    return east_pivot, west_pivot, east_items, west_items


def find_farest(pivot, candidates):
    max_d = 0
    most_point = pivot

    for candidate in candidates:
        cur_d = cal_distance(pivot, candidate)
        if  cur_d > max_d:
            max_d = cur_d
            most_point = candidate
    
    return most_point


def cal_distance(p1, p2):
    return math.sqrt(sum([(v1 - v2) ** 2 for v1, v2 in zip(p1[:-1], p2[:-1])]))


def process_mixed_cluster(cluster):
    """
    in DE operation, use current-to-best to mutate the candidates
    v_i = x_i + F * (x_b - x_i) + F * (x_r1 - x_r2)
    """
    DE_params = {"F": 0.8, "CR": 1.0}
    pos_point = [(idx, item) for idx, item in enumerate(cluster) if item[-1] == 1]
    neg_point = [(idx, item) for idx, item in enumerate(cluster) if item[-1] == 0]

    candidate_l = []
    if len(pos_point) == 1:
        # only 1 pos point in cluster, then mutate all neg points toward to the pos point
        xb = pos_point[0][1]
        R = random.choice(range(len(xb)-1))

        for _, xi in neg_point:
            new_candidate = []
            for i in range(len(xi)-1):
                ri = np.random.uniform(low=0.0, high=1.0, size=1)[0]

                if ri < DE_params["CR"] or i == R:
                    new_candidate.append(xi[i] + DE_params["F"] * (xb[i] - xi[i]))
                else:
                    new_candidate.append(xi[i])
            
            new_candidate.append(1)
            candidate_l.append(np.array(new_candidate))
    else:
        # more than 2 pos points in cluster, then randomly pick 3 points, first 1 is current point, and
        # another two are support points
        for idx1, xb in pos_point:
            R = random.choice(range(len(xb)-1))
            for idx2, xi in enumerate(cluster):
                if not idx1 == idx2:
                    available_points = []
                    for idx3, p in enumerate(cluster):
                        if not idx3 == idx2 and not idx3 == idx1:
                            available_points.append(p)
                    [xr1, xr2] = random.sample(available_points, 2)
                    
                    new_candidate = []
                    # if xr1 and xr2 all negative class, then just use xi
                    if xr1[-1] == 0 and xr2[-1] == 0:
                        for i in range(len(xi)-1):
                            ri = np.random.uniform(low=0.0, high=1.0, size=1)[0]

                            if ri < DE_params["CR"] or i == R:
                                new_candidate.append(xi[i] + DE_params["F"] * (xb[i] - xi[i]))
                            else:
                                new_candidate.append(xi[i])
                    else:
                        for i in range(len(xi)-1):
                            ri = np.random.uniform(low=0.0, high=1.0, size=1)[0]

                            if ri < DE_params["CR"] or i == R:
                                if xr1[-1] == 1:
                                    new_candidate.append(xi[i] + DE_params["F"] * (xb[i] - xi[i]) + DE_params["F"] * (xr1[i] - xr2[i]))
                                else:
                                    new_candidate.append(xi[i] + DE_params["F"] * (xb[i] - xi[i]) + DE_params["F"] * (xr2[i] - xr1[i]))
                            else:
                                new_candidate.append(xi[i])
                    
                    new_candidate.append(1)
                    candidate_l.append(np.array(new_candidate))

    return candidate_l


def process_positive_cluster(cluster):
    """
    in DE operation, use best to mutate the candidates
    v_i = x_b + F * (x_r1 - x_r2)
    """
    DE_params = {"F": 0.8, "CR": 1.0}
    pos_point = [item for item in cluster if item[-1] == 1]
    candidate_l = []

    for idx1 in range(len(pos_point)-2):
        for idx2 in range(idx1+1, len(pos_point)-1):
            for idx3 in range(idx2+1, len(pos_point)):
                [xb, xr1, xr2] = random.sample([idx1, idx2, idx3], 3)
                xb, xr1, xr2 = pos_point[xb], pos_point[xr1], pos_point[xr2]

                R = random.choice(range(len(xb)-1))
                new_candidate = []

                for i in range(len(xb)-1):
                    ri = np.random.uniform(low=0.0, high=1.0, size=1)[0]

                    if ri < DE_params["CR"] or i == R:
                        new_candidate.append(xb[i] + DE_params["F"] * (xr1[i] - xr2[i]))
                    else:
                        new_candidate.append(xb[i])
                
                new_candidate.append(1)
                candidate_l.append(np.array(new_candidate))
    
    return candidate_l


def process_mixed_cluster_extra(cluster):
    """
    in DE operation, use current-to-best-extra to mutate the candidates
    v_i = x_b + F * (x_r1 - x_r2) + F_ex * (x_r3 - x_r4)
    """
    DE_params = {"F": 0.8, "CR": 1.0, "F_xc": 0.1}
    pos_point = [item for item in cluster if item[-1] == 1]

    candidate_l = []
    for xb in pos_point:
        R = random.choice(range(len(xb)-1))

        for xi in cluster:
            if not np.array_equal(xb, xi):
                available_points = []
                for p in cluster:
                    if not np.array_equal(p, xi) and not np.array_equal(p, xb):
                        available_points.append(p)

                for _ in range(20):
                    [xr1, xr2, xr3, xr4] = random.sample(available_points, 4)

                    new_candidate = []

                    for i in range(len(xi)-1):
                        ri = np.random.uniform(low=0.0, high=1.0, size=1)[0]

                        if ri < DE_params["CR"] or i == R:
                            new_candidate.append(xi[i] + DE_params["F"] * (xb[i] - xi[i]) + DE_params["F_xc"] * (xr1[i] - xr2[i]) + DE_params["F_xc"] * (xr3[i] - xr4[i]))
                        else:
                            new_candidate.append(xi[i])

                    new_candidate.append(1)
                    candidate_l.append(new_candidate)
    
    return candidate_l


def RandomProjectionOversampling(X_train, y_train):
    tar = y_train.name
    X_train[tar] = y_train
    X_train.reset_index(inplace=True, drop=True)
    col_names = X_train.columns

    n_data_to_generate = X_train[tar].value_counts()[0] - X_train[tar].value_counts()[1]

    X_train = X_train.to_numpy()
    start_time = time.time()
    res = cluster(X_train, 12, [])
    
    new_data_negative_cluster = []
    new_data_positive_cluster = []
    for c in res:
        if sum([item[-1] for item in c]) > len(c)//2:
            cur_new_data = process_positive_cluster(c)
            new_data_positive_cluster += cur_new_data
        else:
            cur_new_data = process_mixed_cluster(c)
            new_data_negative_cluster += cur_new_data
    
    rt = time.time() - start_time

    if len(new_data_negative_cluster) >= n_data_to_generate - len(new_data_positive_cluster):
        new_data = new_data_positive_cluster + random.sample(new_data_negative_cluster, 
                                                            n_data_to_generate - len(new_data_positive_cluster))
    else:
        extra_data = []
        for c in res:
            cur_extra_data = process_mixed_cluster_extra(c)
            extra_data += cur_extra_data
        
        rest_data_to_generate = n_data_to_generate - len(new_data_positive_cluster) - len(new_data_negative_cluster)
        new_data = new_data_negative_cluster + new_data_positive_cluster + random.sample(extra_data, rest_data_to_generate)
    
    new_data = pd.DataFrame(np.vstack((X_train, np.array(new_data))), columns=col_names)
    
    X_train_new = new_data.iloc[:, :-1]
    y_train_new = new_data.iloc[:, -1]

    return rt, X_train_new, y_train_new


# data_path = f"{os.getcwd()}\\data\\JavaScript_Vulnerability\\"
# datafiles = [f for f in os.listdir(data_path) if f.endswith("csv")]
# df = pd.read_csv(f"{data_path}\\{datafiles[0]}")
# drop_columns = ["name", "longname", "path", "full_repo_path", "line", "column", "endline", "endcolumn"]
# df = df.drop(drop_columns, axis=1)
# # df = df.drop_duplicates()
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X)
# X = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# rt, X_train_new, y_train_new = RandomProjectionOversampling(X_train=X_train, y_train=y_train)
# print(X_train_new)
# print(str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

# data_path = f"{os.path.dirname(os.getcwd())}\\SyntheticData\\data\\Vulnerable_Files\\moodle-2_0_0-metrics.arff"
# data = arff.loadarff(data_path)
# df = pd.DataFrame(data[0])
# df['IsVulnerable'] = df['IsVulnerable'].astype('str')
# d = {'b\'yes\'': 1, 'b\'no\'': 0}
# df['IsVulnerable'] = df['IsVulnerable'].astype(str).map(d).fillna(df['IsVulnerable'])
# df = df.drop_duplicates()
# df.reset_index(inplace=True, drop=True)
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# rs = random.randint(0, 100000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)
# rt, X_train_new, y_train_new = RandomProjectionOversampling(X_train=X_train, y_train=y_train)