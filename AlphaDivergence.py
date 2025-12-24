import numpy as np
from surprise import PredictionImpossible
from six import iteritems
from surprise import AlgoBase
import heapq
from statistics import median
from scipy.stats import entropy
from statistics import stdev
from surprise.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from scipy.stats import dirichlet
import collections
from tqdm import tqdm
import math
from surprise import accuracy
# from surprise.prediction_algorithms.predictions import Prediction
from collections import OrderedDict


class mySymmetricAlgo(AlgoBase):
    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items   # |U|or|I|
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users   # |U|or|I|
        self.xr = self.trainset.ur if ub else self.trainset.ir   # user ratings or item ratings
        self.yr = self.trainset.ir if ub else self.trainset.ur   # user ratings or item ratings

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class AlphaDivergenceSimilarity(mySymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True,**kwargs):
        
        mySymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)
        self.k = k
        self.min_k = min_k
        # self.alpha = alpha
        self.beta = 0.5

    def fit(self, trainset):
        mySymmetricAlgo.fit(self, trainset)
        print('Computing item similarity matrix...')
        item_similarity = np.zeros((self.trainset.n_items, self.trainset.n_items), dtype=np.double)
        
        # Calculate empirical distributions
        empirical_distributions = {}
        for _, _, rating in self.trainset.all_ratings():
            if rating not in empirical_distributions:
                empirical_distributions[rating] = 1
            else:
                empirical_distributions[rating] += 1
        
        # Normalize empirical distributions
        total_ratings = sum(empirical_distributions.values())
        for rating in empirical_distributions:
            empirical_distributions[rating] /= total_ratings
            
        self.means = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.means[x] = np.mean([r for (_, r) in ratings])
        # print(self.means)

        self.itemMaxlen = max([len(rates) for _, rates in self.trainset.ir.items()]) 
        self.userMaxlen = max([len(rates) for _, rates in self.trainset.ur.items()])

        self.itemmeans = np.zeros(self.trainset.n_items)
        self.item_max_rating = np.zeros(self.trainset.n_items)
        self.item_min_rating = np.zeros(self.trainset.n_items)
        for x, ratings in iteritems(self.trainset.ir):
            self.itemmeans[x] = np.mean([r for (_, r) in ratings])
            self.item_max_rating[x] = max([r for (_, r) in ratings])
            self.item_min_rating[x] = min([r for (_, r) in ratings])
            
        self.itemstd = np.zeros(self.trainset.n_items)
        for x, ratings in iteritems(self.trainset.ir):
            self.itemstd[x] = np.std([r for (_, r) in ratings])

        # print(self.itemmeans)

        self.median = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.median[x] = median([r for (_, r) in ratings])

        self.std = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.std[x] = np.std([r for (_, r) in ratings])

        self.medianvalue = (trainset.rating_scale[1] + trainset.rating_scale[0]) / 2.0
        self.maxminesmin = (trainset.rating_scale[1] - trainset.rating_scale[0]) * 1.0
        
        all_ratings = [r for (_, _, r) in self.trainset.all_ratings()]
        self.med_plus = np.median([r for r in all_ratings if r >= self.medianvalue])
        self.med_minus = np.median([r for r in all_ratings if r < self.medianvalue])

        self.medstd = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            # self.medstd[x] = math.sqrt(sum(pow(x - self.median[x], 2) for (_, x) in ratings) / len(ratings))
            # self.medstd[x] = math.sqrt(np.sum(np.power([x - self.median[x] for (_, x) in ratings], 2)))
            self.medstd[x] = math.sqrt(np.sum(np.power([x - self.medianvalue for (_, x) in ratings], 2)))
        
        pbar = tqdm(total=(self.trainset.n_items * self.trainset.n_items))
        for itemi, ratesi in self.trainset.ir.items():
            for itemj, ratesj in self.trainset.ir.items():
           
                # Calculate empirical distributions for item i and j
                empirical_distributioni = np.zeros(len(empirical_distributions), dtype=np.double)
                empirical_distributionj = np.zeros(len(empirical_distributions), dtype=np.double)
                
                for _, ratei in ratesi:
                    empirical_distributioni[list(empirical_distributions.keys()).index(ratei)] += 1
                for _, ratej in ratesj:
                    empirical_distributionj[list(empirical_distributions.keys()).index(ratej)] += 1
                
                # Normalize empirical distributions for item i and j
                empirical_distributioni /= len(ratesi)
                empirical_distributionj /= len(ratesj)
                
                alpha = 5

                # Calculate α-divergence
                divergence = 0
                p = empirical_distributioni
                q = empirical_distributionj
                for h in range(len(p)):
                    if not (p[h] and q[h]):
                        term = alpha * p[h] + (1 - alpha) * q[h]
                    else:
                        term = alpha * p[h] + (1 - alpha) * q[h] - (p[h]** alpha) * (q[h]**(1 - alpha))
                    divergence += term
                divergence /= (alpha * (1 - alpha))
                # item_similarity[itemi, itemj] = math.exp(-divergence) 
                ############################################################################################
                item_similarity[itemi, itemj]  = min(1.0, math.exp(-divergence))
                pbar.update(1)
        pbar.close()
        

        print('Computing user similarity matrix...')
        mySimilarity = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        
        
        pbar = tqdm(total=self.trainset.n_users * self.trainset.n_users)
        for useri, ratesi in self.trainset.ur.items():
            if useri not in self.testusers:
                pbar.update(trainset.n_users)
                continue
            for userj, ratesj in self.trainset.ur.items():
                PSS1 = 1
                PSS2 = 0
                Commonitems = 0 # 数量
                Gower = 0
                d = np.zeros((self.trainset.n_items), dtype=np.double)
                for itemi, ratei in ratesi:
                    for itemj, ratej in ratesj:
                        if itemi == itemj:
                            Commonitems += 1
                            rk = self.item_max_rating[itemi] - self.item_min_rating[itemi]
                            # d[itemi] = (1 - math.fabs(ratei - ratej) / (1 + rk)) if (ratei - ratej == rk) else (1 - math.fabs(ratei - ratej) / rk)
                            d[itemi] = (1 - math.fabs(ratei - ratej) / rk) 
                            Gower += d[itemi]
                            
                        if (ratei >= self.medianvalue and ratej >= self.medianvalue) or (ratei <= self.medianvalue and ratej <= self.medianvalue):
                            Proximity = ((math.fabs(ratei - ratej) - (self.med_plus + self.med_minus) / 2) / self.maxminesmin) ** 2
                            Significance = math.exp(-1 / ((math.fabs(ratei - self.medianvalue) +1) * (math.fabs(ratej - self.medianvalue) + 1)))
                            # Significance = math.exp(-1 / ((math.fabs(ratei - self.medianvalue) +1) * (math.fabs(ratej - self.medianvalue) + 1)))
                            Singularity = math.log10(2 + ((ratei + ratej) / 2 - (self.itemmeans[itemi] + self.itemmeans[itemj]) / 2) ** 2) if (ratei > self.itemmeans[itemi] and ratej > self.itemmeans[itemj]) or (ratei < self.itemmeans[itemi] and ratej < self.itemmeans[itemj]) else 0.3010
                          
                        else:
                            if math.fabs(ratei - ratej) > self.medianvalue:
                                delta = 0.5
                            # elif math.fabs(ratei - ratej) == self.medianvalue:
                            #     delta = 0.5
                            else:
                                delta = 0.25
                            Proximity = delta * (self.maxminesmin / math.fabs(ratei - ratej)) ** 2
                            # Proximity = delta * (1 / (self.maxminesmin * math.fabs(ratei - ratej))) ** 2
                            Significance = 1 / ((math.fabs(ratei - self.medianvalue) +1) * (math.fabs(ratej - self.medianvalue) + 1))
                            # Significance = math.exp(-(math.fabs(ratei - self.medianvalue)) * (math.fabs(ratej - self.medianvalue)))
                            Singularity = math.log10(2 + ((ratei + ratej) / 2 - (self.itemmeans[itemi] + self.itemmeans[itemj]) / 2) ** 2) if (ratei > self.itemmeans[itemi] and ratej > self.itemmeans[itemj]) or (ratei < self.itemmeans[itemi] and ratej < self.itemmeans[itemj]) else 0.3010
                    

                        # Proximity = 1 - (1 + math.exp(-math.fabs(ratei - ratej))) ** -1
                        # Significance = (1 + math.exp(
                        #     -1 * math.fabs(ratei - self.medianvalue) * math.fabs(ratej - self.medianvalue))) ** -1
                        # Singularity = 1 - (1 + math.exp(-math.fabs(
                        #     0.5 * (ratei + ratej) - 0.5 * (self.itemmeans[itemi] + self.itemmeans[itemj])))) ** -1
                        
                        # Antipopularity = (2 - (1 + math.exp(-len(self.trainset.ir[itemi]) / self.itemMaxlen)) ** -1) * (
                        #             2 - (1 + math.exp(-len(self.trainset.ir[itemj]) / self.itemMaxlen)) ** -1)
                        Antiprominent = (2 - (1 + math.exp(-len(self.trainset.ur[userj]) / self.userMaxlen)) ** -1)
                        
                        
                        if itemi == itemj:
                            PSS1 *= (1 + Proximity * Significance * Singularity * Antiprominent * item_similarity[itemi, itemj])
                        else:
                            PSS2 += Proximity * Significance * Singularity * Antiprominent * item_similarity[itemi, itemj]

                PSS2 = (1 + PSS2 / (len(ratesi) * len(ratesj) - Commonitems)) if (len(ratesi) * len(
                    ratesj) - Commonitems) != 0 else 1
                PSS = PSS1 * PSS2
                
                URP = math.exp((-1) * math.fabs(self.means[useri] - self.means[userj]))*math.exp((-1) * math.fabs(
                    self.std[useri] - self.std[userj]))
                
                # URP = math.exp((-1) * math.fabs(self.means[useri] - self.means[userj] * math.fabs(self.std[useri] - self.std[userj])))
                # URP = (1 - (math.exp((-1) * math.fabs(self.means[useri] - self.means[userj]))) ** (-1)) * (1 - ( math.exp((-1) * math.fabs(
                    # self.std[useri] - self.std[userj]))) ** (-1))
                wgu = (1 + math.exp((-1) * (Commonitems ** 2 ) / (len(ratesi) + len(ratesj)))) ** (-1) # 加入用户权重Sorgenfrei加号
                Gower = Gower / Commonitems if Commonitems != 0 else 0
                mySimilarity[useri, userj] = URP * PSS * wgu * (1 + math.exp((-1) * Gower)) ** (-1)
                # mySimilarity[useri, userj] = URP * PSS * wgu # 去掉高尔系数
                # mySimilarity[useri, userj] = URP * PSS * (1 + math.exp((-1) * Gower)) ** (-1) #去掉Sorgenfrei权重
                
                pbar.update(1)
        pbar.close()
        self.sim = mySimilarity
    
    
        # print(1)
        # print(mySimilarity)
        print(self.sim)
        print('Done computing user similarity matrix.')

        return self


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):  # both know
            raise PredictionImpossible('User and/or item is unknown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])
                actual_k += 1
        

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        ############################################################################################打印相似度
        # print(f"用户 {u} 的Top-{self.k}邻居相似度: {[sim for _, sim, _ in k_neighbors]}")
        # print(details)
        return est, details

    def nspCalc(self): # 惩罚邻居相似性低的预测，提升鲁棒性
        sp = 0
        total = 0

        for user in self.testusers:
            for item in self.trainset.ir.keys():
                total += 1
                x, y = user, item
                neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
                if not neighbors:  # empty list
                    continue
                k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])
                sum_sim = sum_ratings = actual_k = 0
                for (nb, sim, r) in k_neighbors:
                    if sim > 0:
                        sum_sim += sim
                        sum_ratings += sim * (r - self.means[nb])
                        actual_k += 1

                if actual_k < self.min_k:
                    continue
                if sum_sim == 0:
                    continue
                sp += 1
        ###################################################################################
        return sp/total if total != 0 else 0

    def test2(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    clip=False,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions

