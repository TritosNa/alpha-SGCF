from math import cos, log, log2
from surprise import Dataset, Trainset
from surprise.model_selection import train_test_split
from time import sleep
import numpy as np
import random
import pandas as pd
from surprise import Reader
from collections import OrderedDict
from collections import defaultdict
from surprise import accuracy
import networkx as nx
# import klcore
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest

def create_filmtrust_dataset(seed=19):
    random.seed(seed)
    percentage = 1  # preserve 'percentage' ratings.
    upItem = 5
    data = pd.read_pickle('filmtrust.pkl')
    allratings = [tuple([int(x[0]), int(x[1]), x[2], int(x[3])]) for x in
                  data[['userID', 'musicID', 'rating', 'ex']].values]
    ####将评分转换至1-5
    # allratings = [tuple([int(x[0]), int(x[1]), max(1, min(5, round(1 + (x[2] - 0.5) * (4 / 3.5)))), int(x[3])]) for x in
    #               data[['userID', 'musicID', 'rating', 'ex']].values]
    userdct = {}
    for x in data[['userID', 'musicID', 'rating', 'ex']].values:
        if x[0] not in userdct.keys():
            userdct[x[0]] = 1
        else:
            userdct[x[0]] += 1
    SocialNet = nx.DiGraph()
    # data = pd.read_pickle('Filmtrust.pkl')
    # allusers = [int(x[0]) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
    # for userID in allusers:
    #     SocialNet.add_node(userID)
    edges = np.load('filmtrustEdges.npy', allow_pickle=True).tolist()
    edges = [[u,v] for u,v,w in edges]
    SocialNet.add_edges_from(edges)
    testusers = []
    for user in SocialNet.nodes:
        if user in userdct.keys():
            #########################################################################################################
            if userdct[user] <= upItem:  # cold user
                testusers.append(user)
    print(len(testusers))
    # testusers = sorted(list(SocialNet.nodes), key=lambda x: (SocialNet.in_degree(x) + SocialNet.out_degree(x)), reverse=False)[:256]
    allratings = sorted(allratings, key=lambda x: (int(x[0]), int(x[3])))
    allratingsDict = {}
    for element in allratings:
        user = element[0]
        item = element[1]
        rating = element[2]
        time = element[3]
        # # 这里做整数映射
        # rating = round(1 + (rating - 0.5) * (4 / 3.5))
        # rating = max(1, min(5, rating))  # 保证结果在1-5之间
        if user not in allratingsDict.keys():
            allratingsDict[user] = []
        allratingsDict[user].append((item, rating, time))
    allratingsDict = OrderedDict(sorted(allratingsDict.items()))
    #  FIXME: split 20% test users from whom items will be removed 80%.
    #  FIXME: Warning...
    #  FIXME: Warning...
    
    max_userid = -1
    max_itemid = -1
    for u, i, *_ in allratings:  # 使用 *_ 忽略其他字段（如评分、时间戳）
        if int(u) > max_userid:
            max_userid = int(u)
        if int(i) > max_itemid:
            max_itemid = int(i)
            
    trainDict = {'userID': [], 'itemID': [], 'rating': []}
    testDict = {'userID': [], 'itemID': [], 'rating': []}
    for user in allratingsDict.keys():
        if user in testusers:  # test users
            for entry in allratingsDict[user]:
                #################################################################################################
                # 如果用户已经被划分为测试集，则根据一定的概率将他们的数据加入训练集
                if testDict['userID'].count(user) < 1:# 如果测试用户未被加入测试集，那么加入
                # if random.uniform(0, 1) > percentage:
                    testDict['userID'].append(user)
                    testDict['itemID'].append(entry[0])
                    testDict['rating'].append(entry[1])
                else:# 否则，按一定概率划分训练集和测试集
                    if random.uniform(0, 1) <= percentage:
                        trainDict['userID'].append(user)
                        trainDict['itemID'].append(entry[0])
                        trainDict['rating'].append(entry[1])
        else:
            for entry in allratingsDict[user]:
                if random.uniform(0, 1) <= percentage:
                    trainDict['userID'].append(user)
                    trainDict['itemID'].append(entry[0])
                    trainDict['rating'].append(entry[1])
                    
    df = pd.DataFrame(trainDict)
    num_users = df['userID'].nunique()
    num_items = df['itemID'].nunique()
    num_ratings = df['rating'].nunique

    # print(f"总用户数: {num_users}")
    # print(f"总项目数: {num_items}")
    # print(f"总评分数: {num_ratings}")
    # num_users = df['userID'].nunique()
    # num_items = df['itemID'].nunique()
    # m, n = df.shape
    # print(df.shape)
    # print(f"稀疏性: {m/(num_users * num_items)}")
    
    trainset = Dataset.load_from_df(pd.DataFrame(trainDict)[['userID', 'itemID', 'rating']],
                                    Reader(rating_scale=(0.5, 4))).build_full_trainset()
    testset = Dataset.load_from_df(pd.DataFrame(testDict)[['userID', 'itemID', 'rating']],
                                   Reader(rating_scale=(0.5, 4))).build_full_trainset().build_testset()
    return trainset, testset, max_userid, max_itemid

def createdataset(seed=19, name='ml-100k'):
    print('dataset is ' + name)
    random.seed(seed)
    percentage = 0.2  # preserve 'percentage' ratings.
    testpercentage = 0.2
    upItem = 5
    if name == 'yahoo':
        percentage = 0.2
        data = pd.read_pickle('yahoo.pkl')
        allratings = [tuple(x) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
    elif name == 'filmtrust':
        print('Shortcut!')
        ###########################################################################################
        return create_filmtrust_dataset(seed=seed)
    elif name == 'ciao':
        percentage = 0.2
        data = pd.read_pickle('ciao.pkl')
        allratings = [tuple([int(x[0]), int(x[1]), x[2], int(x[3])]) for x in
                                 data[['userID', 'musicID', 'rating', 'ex']].values]
    elif name == 'ml-ls':
        percentage = 0.2
        data = pd.read_pickle('ml-ls.pkl')
        allratings = [tuple([int(x[0]), int(x[1]), x[2], int(x[3])]) for x in
                                    data[['userID', 'musicID', 'rating', 'ex']].values]
    elif name == 'epinions':
        percentage = 0.2
        data = pd.read_pickle('epinions.pkl')
        allratings = [tuple([int(x[0]), int(x[1]), x[2], int(x[3])]) for x in
                                 data[['userID', 'musicID', 'rating', 'ex']].values]
    # elif name == 'example':
    #     percentage = 0.2
    #     data = pd.read_pickle('example.pkl')
    #     allratings = [tuple(x) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
    else:
        data = Dataset.load_builtin(name)
        allratings = data.raw_ratings
    allratings = sorted(allratings, key=lambda x: (int(x[0]), int(x[3])))
    allratingsDict = {}
    for element in allratings:
        user = element[0]
        item = element[1]
        rating = element[2]
        time = element[3]
        if user not in allratingsDict.keys():
            allratingsDict[user] = []
        allratingsDict[user].append((item, rating, time))
    allratingsDict = OrderedDict(sorted(allratingsDict.items()))
    #  FIXME: split 20% test users from whom items will be removed 80%.
    #  FIXME: Warning...
    #  FIXME: Warning...
    
    max_userid = -1
    max_itemid = -1
    for u, i, *_ in allratings:  # 使用 *_ 忽略其他字段（如评分、时间戳）
        if int(u) > max_userid:
            max_userid = int(u)
        if int(i) > max_itemid:
            max_itemid = int(i)

    # print(f"最大用户ID: {max_userid}, 最大物品ID: {max_itemid}")
    
    trainDict = {'userID': [], 'itemID': [], 'rating': []}
    testDict = {'userID': [], 'itemID': [], 'rating': []}
    pbar = tqdm(total=len(allratingsDict.keys()))
    for user in allratingsDict.keys():
        if random.uniform(0, 1) <= testpercentage:  # test users
            for entry in allratingsDict[user]:
                if random.uniform(0, 1) <= percentage and trainDict['userID'].count(user) <= upItem:  # training data
                    trainDict['userID'].append(user)
                    trainDict['itemID'].append(entry[0])
                    trainDict['rating'].append(entry[1])
                else:
                    if testDict['userID'].count(user) < 20:
                        if name not in ['Epinions']:
                            testDict['userID'].append(user)
                            testDict['itemID'].append(entry[0])
                            testDict['rating'].append(entry[1])
                        else:
                            if random.uniform(0, 1) <= 0.1:
                                testDict['userID'].append(user)
                                testDict['itemID'].append(entry[0])
                                testDict['rating'].append(entry[1])
        else:
            for entry in allratingsDict[user]:
                if random.uniform(0, 1) <= percentage:
                    trainDict['userID'].append(user)
                    trainDict['itemID'].append(entry[0])
                    trainDict['rating'].append(entry[1])
        pbar.update(1)
    pbar.close()
    
    df = pd.DataFrame(trainDict)
    num_users = df['userID'].nunique()
    num_items = df['itemID'].nunique()
    num_ratings = df['rating'].nunique

    # print(f"总用户数: {num_users}")
    # print(f"总项目数: {num_items}")
    # print(f"总评分数: {num_ratings}")
    # num_users = df['userID'].nunique()
    # num_items = df['itemID'].nunique()
    # m, n = df.shape
    # print(df.shape)
    # print(f"稀疏性: {m/(num_users * num_items)}")
    
    trainset = Dataset.load_from_df(pd.DataFrame(trainDict)[['userID', 'itemID', 'rating']],
                                    Reader(rating_scale=(1, 5))).build_full_trainset()
    testset = Dataset.load_from_df(pd.DataFrame(testDict)[['userID', 'itemID', 'rating']],
                                   Reader(rating_scale=(1, 5))).build_full_trainset().build_testset()
    return trainset, testset, max_userid, max_itemid

# def createdataset(seed=19, name='ml-100k'):
#     print('dataset is ' + name)
#     random.seed(seed)

#     def process_cold_start_split(allratings, rating_transform=False, seed=19):
#         from collections import defaultdict
#         random.seed(seed)
        
#         # # 第一步：随机移除80%的评分 效果非常差
#         # remove_num = int(0.8 * len(allratings))
#         # allratings = random.sample(allratings, len(allratings) - remove_num)

#         if rating_transform:
#             # 将filmtrust评分从0.5~4转换成1~5整数
#             allratings = [
#                 (int(u), int(i), max(1, min(5, round(1 + (r - 0.5) * (4 / 3.5)))), int(t))
#                 for u, i, r, t in allratings
#             ]
#         else:
#             allratings = [
#                 (int(u), int(i), int(r), int(t)) if not isinstance(r, str) else (int(u), int(i), int(float(r)), int(t))
#                 for u, i, r, t in allratings
#             ]

#         # 将所有评分按用户分组，并按评分时间排序，保证后续取样时有时间先后顺序
#         allratings = sorted(allratings, key=lambda x: (x[0], x[3]))
#         user_ratings = defaultdict(list)
#         # item_ratings = defaultdict(list)
#         for u, i, r, t in allratings:
#             user_ratings[u].append((i, r, t))
#         max_userid = -1
#         max_itemid = -1
#         for u, i, *_ in allratings:  # 使用 *_ 忽略其他字段（如评分、时间戳）
#             if u > max_userid:
#                 max_userid = u
#             if i > max_itemid:
#                 max_itemid = i

#         # print(f"最大用户ID: {max_userid}, 最大物品ID: {max_itemid}")

#         # 将所有用户的 ID 提取成一个列表
#         user_list = list(user_ratings.keys())
#         random.shuffle(user_list)
#         cold_users = set(user_list[:int(0.2 * len(user_list))]) # 随机选择20%用户作为冷启动用户

#         train_dict = {'userID': [], 'itemID': [], 'rating': []}
#         test_dict = {'userID': [], 'itemID': [], 'rating': []}

#         for u in user_ratings:
#             upItem = 5
#             items = user_ratings[u] # 该用户的评分列表
#             if u in cold_users:
            
#                 # if len(items) <= upItem:
#                 #     for i, r, _ in items:
#                 #         train_dict['userID'].append(u)
#                 #         train_dict['itemID'].append(i)
#                 #         train_dict['rating'].append(r)
#                 # elif len(items) < 25:
#                 #     selected = random.sample(items, len(items))
#                 #     for i, r, _ in selected[:5]:
#                 #         train_dict['userID'].append(u)
#                 #         train_dict['itemID'].append(i)
#                 #         train_dict['rating'].append(r)
#                 #     for i, r, _ in selected[5:len(items)]:
#                 #         test_dict['userID'].append(u)
#                 #         test_dict['itemID'].append(i)
#                 #         test_dict['rating'].append(r)        
#                 ###########################################################
#                 if len(items) <= upItem:
#                     for i, r, _ in items:
#                         train_dict['userID'].append(u)
#                         train_dict['itemID'].append(i)
#                         train_dict['rating'].append(r)
#                 else:
#                     num_train = max(1, int(0.2 * len(items)))  # 保留20%评分作为训练，至少1条
#                     # selected = random.sample(items, len(items))
#                     selected = random.sample(items, num_train)
#                     for i, r, _ in selected[:upItem]:
#                         train_dict['userID'].append(u)
#                         train_dict['itemID'].append(i)
#                         train_dict['rating'].append(r)
#                     if len(items) <= 20:
#                         for i, r, _ in selected[upItem:]:
#                             test_dict['userID'].append(u)
#                             test_dict['itemID'].append(i)
#                             test_dict['rating'].append(r)
#                     else:
#                         for i, r, _ in selected[upItem:20]:
#                         # if random.uniform(0, 1) <= 0.1:
#                             test_dict['userID'].append(u)
#                             test_dict['itemID'].append(i)
#                             test_dict['rating'].append(r)    

                
#             else:
#                 for i, r, _ in items:
#                     train_dict['userID'].append(u)
#                     train_dict['itemID'].append(i)
#                     train_dict['rating'].append(r)

#         trainset = Dataset.load_from_df(pd.DataFrame(train_dict)[['userID', 'itemID', 'rating']],
#                                         Reader(rating_scale=(1, 5))).build_full_trainset()
#         testset = Dataset.load_from_df(pd.DataFrame(test_dict)[['userID', 'itemID', 'rating']],
#                                        Reader(rating_scale=(1, 5))).build_full_trainset().build_testset()
#         return trainset, testset, max_userid, max_itemid
    

#     if name == 'yahoo':
#         data = pd.read_pickle('yahoo.pkl')
#         allratings = [tuple(x) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
#     elif name == 'filmtrust':
#         data = pd.read_pickle('filmtrust.pkl')
#         allratings = [tuple(x) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
#         return process_cold_start_split(allratings, rating_transform=True, seed=seed)
#     elif name == 'ciao':
#         data = pd.read_pickle('ciao.pkl')
#         allratings = [tuple(x) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
#     elif name == 'ml-ls':
#         data = pd.read_pickle('ml-ls.pkl')
#         allratings = [tuple(x) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
#     elif name == 'epinions':
#         data = pd.read_pickle('epinions.pkl')
#         allratings = [tuple(x) for x in data[['userID', 'musicID', 'rating', 'ex']].values]
#     else:
#         data = Dataset.load_builtin(name)
#         allratings = data.raw_ratings

#     return process_cold_start_split(allratings, rating_transform=False, seed=seed)


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    pbar = tqdm(total=len(user_est_true.items()))
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items 实际喜欢的项目集Ia
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k 推荐的项目集Ip
        n_rec_k = len(user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in user_ratings[:k])

        ##############################################################################################
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        pbar.update(1)
    pbar.close()
    return precisions, recalls


def nppCalc(predictions): # 归一化精确预测率，衡量模型的高精度预测能力
    npp = 0
    for uid, _, true_r, est, _ in predictions:
        ########################################################################################################################
        # if np.fabs(est - true_r) <= 0.5: # 效果也不错
        if np.fabs(est - true_r) <= 0.2:
            npp += 1
    return npp/len(predictions) 


# def calculate_diversity(predictions, matrix, k=10):
    
#     sim_matrix = cosine_similarity(matrix.T)
    
#     user_recommendations = defaultdict(list)
#     for uid, iid, _, est, _ in predictions:
#         user_recommendations[uid].append((iid, est))   
                 
#     total_diversity = 0.0
#     valid_users = 0
    
#     for uid, user_ratings in user_recommendations.items():
#         # 获取Top-K推荐项目集I_p(u)
#         user_ratings.sort(key=lambda x: x[1], reverse=True)
#         I_p = [int(iid) for iid, _ in user_ratings[:k]]
#         onelist = [1]*len(I_p)
#         index = [x - y for x,y in zip(I_p, onelist)]
#         sim_matrixuid = sim_matrix[np.ix_(index, index)]
        
#         if len(I_p) < 2:  # 至少需要2个物品计算多样性
#             continue
        
#         np.fill_diagonal(sim_matrixuid, 0)  # 忽略自身相似度
        
#         # 单个用户多样性公式(19)
#         n_pairs = len(I_p) * (len(I_p) - 1)
#         diversity_u = np.sum(1 - sim_matrixuid) / n_pairs
#         total_diversity += diversity_u
#         valid_users += 1
    
#     # 总体多样性公式(18)
#     return total_diversity / valid_users if valid_users > 0 else 0

def calculate_novelty(predictions, k=10):
    # Step 1: 统计物品被推荐次数，并按用户分组（保留原始预测）
    item_rec_counts = defaultdict(int)
    user_recommendations = defaultdict(list)
    
    for uid, iid, _, est, _ in predictions:
        user_recommendations[uid].append((iid, est))
        item_rec_counts[iid] += 1
    
    # Step 2: 对每个用户只保留Top-K个评分（按est降序）
    for uid in user_recommendations:
        user_recommendations[uid] = nlargest(k, user_recommendations[uid], key=lambda x: x[1])
    
    # Step 3: 重新统计Top-K后的物品推荐次数（重要！）
    item_rec_counts_topk = defaultdict(int)
    for uid, items in user_recommendations.items():
        for iid, _ in items:
            item_rec_counts_topk[iid] += 1
    
    # Step 4: 计算p(i|s)概率
    total_recs = sum(item_rec_counts_topk.values())
    p_i_s = {iid: count/total_recs for iid, count in item_rec_counts_topk.items()}
    
    # Step 5: 计算总新颖性
    total_novelty = 0.0
    for iid, prob in p_i_s.items():
        if prob > 0:
            total_novelty += -prob * log2(prob)   
    return total_novelty

    
def calculate_hr(predictions, k=10, threshold=4):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    hr_numerator = 0
    hr_denominator = 0
    total_ndcg = 0.0
    valid_users = 0

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        I_a = [iid for (iid, true_r) in user_ratings if true_r >= threshold]
        I_p = [iid for (iid, _) in user_ratings[:k]]
        
        hr_numerator += len(set(I_p) & set(I_a))
        hr_denominator += len(I_a)
        
        dcg = sum((2**(1 if true_r>=threshold else 0)-1)/log2(idx+1) 
                 for idx, (_, true_r) in enumerate(user_ratings[:k],1))
        idcg = sum((2**1-1)/log2(idx+1) 
                  for idx, (_, true_r) in enumerate(
                      sorted(user_ratings, key=lambda x: x[1], reverse=True)[:k],1))
        
        total_ndcg += dcg/idcg if idcg>0 else 0
        valid_users += 1
        
    hr = hr_numerator/hr_denominator if hr_denominator>0 else 0
    ndcg = total_ndcg/valid_users if valid_users>0 else 0
    return hr, ndcg
def create_testusers(trainset, testset):
    testusers = []
    for rawid, _, _ in testset:
        try:
            testusers.append(trainset.to_inner_uid(rawid))
        except:
            pass
        continue
    testusers = list(set(testusers))
    return testusers

# def u_i_matrix(trainset, Nu, Ni):

#         matrix = np.zeros((Nu,Ni))
#         for u, i, r in trainset.all_ratings():
#             matrix[u, i] = r  # 使用内部ID
#         return matrix


class CalMetric(object):
    def __init__(self):
        self.ndcgList = []
        self.hrList = []
        self.noveltyList = []
        self.coverageList = []
        self.diversityList = []
        self.maeList = []
        self.rmseList = []
        self.preList = []
        self.recList = []
        self.f1List = []
        self.totalndcg = 0
        self.totalhr = 0
        self.totalnovelty = 0
        self.totalcoverage = 0
        self.totaldiversity = 0
        self.totalmae = 0
        self.totalrmse = 0
        self.totalpre = 0
        self.totalrec = 0
        self.totalf1 = 0
        self.totalnpp = 0
        self.totalnsp = 0
        self.metricTensor = {}
        self.nppnsp = {}
        self.sim_options = {'name': 'cosine',
                       'user_based': True,  # compute similarities between users
                       'min_support': 0
                       }
    def Curvecvcalculate(self, model, fold=5, neighbours=None):
        print('Using Curve cross-validation')
        self.metricTensor = {'ndcg': np.zeros((fold, len(neighbours))), 'hr': np.zeros((fold, len(neighbours))), 'novelty': np.zeros((fold, len(neighbours))), 'coverage': np.zeros((fold, len(neighbours))), 'diversity': np.zeros((fold, len(neighbours))), 'mae': np.zeros((fold, len(neighbours))), 'rmse': np.zeros((fold, len(neighbours))), 'pre': np.zeros((fold, len(neighbours))), 'rec': np.zeros((fold, len(neighbours))), 'f1': np.zeros((fold, len(neighbours)))}
        # self.metricTensor = {'pre': np.zeros((fold, len(neighbours))), 'rec': np.zeros((fold, len(neighbours))), 'f1': np.zeros((fold, len(neighbours)))}
        self.nppnsp = {'npp': np.zeros((fold, len(neighbours))), 'nsp': np.zeros((fold, len(neighbours)))}
        for i in range(fold):
            print('Here is fold '+str(i+1))
            # trainset, testset, Nu, Ni = createdataset(seed=19+i, name='epinions') # 太大没法跑
            # trainset, testset, Nu, Ni = createdataset(seed=19+i, name='ciao')
            # trainset, testset, Nu, Ni = createdataset(seed=19+i, name='filmtrust')
            # trainset, testset, Nu, Ni = createdataset(seed=19+i, name='ml-ls')
            trainset, testset, Nu, Ni = createdataset(seed=19+i, name='ml-1m')
            # trainset, testset, Nu, Ni = createdataset(seed=19+i, name='ml-100k')
            # trainset, testset, Nu, Ni = createdataset(seed=19+i, name='yahoo')
            testusers = []
            for rawid, _, _ in testset:
                try:
                    testusers.append(trainset.to_inner_uid(rawid))
                except:
                    pass
                continue
            testusers = list(set(testusers))
            # matrix = u_i_matrix(trainset,Nu,Ni)
            #####################################################################################################################
            algo = model(2, 1, sim_options=self.sim_options, verbose=True)
            algo.testusers = testusers
            algo.fit(trainset)
            for j, neighbour in enumerate(neighbours):
                print('Here neighbour is '+str(neighbour))
                print('Here is ' + str(j+1) + '/' + str(len(neighbours)))
                algo.k = neighbour
                print(algo.k)
                ############################################################################################################
                predictions = algo.test(testset)
                # print(predictions)
                predictions2 = algo.test2(testset) # 用于计算npp指标
                # pnsp = algo.nspCalc()
                pnpp = nppCalc(predictions2)
                #########################################################################################################################
                # precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
                # p = sum(prec for prec in precisions.values()) / len(precisions)
                # r = sum(rec for rec in recalls.values()) / len(recalls)
                # f = 2 * p * r / (p + r)
                #########################################################################################################################
                hr , ndcg= calculate_hr(predictions, k=10, threshold=4)
                novelty = calculate_novelty(predictions, k=10)
                # diversity = calculate_diversity(predictions, matrix, k=10)
                self.metricTensor['ndcg'][i][j] = ndcg
                self.metricTensor['hr'][i][j] = hr
                self.metricTensor['novelty'][i][j] = novelty
                # self.metricTensor['coverage'][i][j] = coverage
                # self.metricTensor['diversity'][i][j] = diversity
                self.nppnsp['npp'][i][j] = pnpp
                # self.nppnsp['nsp'][i][j] = pnsp
                self.metricTensor['mae'][i][j] = accuracy.mae(predictions, verbose=False) # 输出到终端
                self.metricTensor['rmse'][i][j] = accuracy.rmse(predictions, verbose=False) # , verbose=False 不输出到终端
                # self.metricTensor['pre'][i][j] = p
                # self.metricTensor['rec'][i][j] = r
                # self.metricTensor['f1'][i][j] = f
        # return self.metricTensor['pre'].mean(axis=0), self.metricTensor['rec'].mean(axis=0), self.metricTensor['f1'].mean(axis=0)    
        return self.metricTensor['ndcg'].mean(axis=0), self.metricTensor['hr'].mean(axis=0), self.metricTensor['novelty'].mean(axis=0), self.metricTensor['mae'].mean(axis=0), self.metricTensor['rmse'].mean(axis=0), self.nppnsp['npp'].mean(axis=0)
    def clearmetric(self):
        self.totalmae = 0
        self.totalrmse = 0
        self.totalpre = 0
        self.totalrec = 0
        self.totalf1 = 0
        self.sim_options = {'name': 'cosine',
                            'user_based': True,  # compute similarities between users
                            'min_support': 0
                            }


# if __name__ == '__main__':
#     create_filmtrust_dataset()