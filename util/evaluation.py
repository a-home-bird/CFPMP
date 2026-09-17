import math
from collections import defaultdict
import pickle

class Metric(object):
    def __init__(self):
        pass
    
    @staticmethod
    def P(origin, res):
        p_hit = [[],[],[],[],[]]
        for user in origin:
            items = list(origin[user].keys())
            for i in range(5):
                predicted = [item[0] for item in res[user][:i+1]]

                p_hit[i].append(len(set(items).intersection(set(predicted)))/(i+1))
        p_result = []
        for i in range(5):
            p_result.append(sum(p_hit[i])/len(p_hit[i]))
        return p_result
    
    
    @staticmethod
    def P_sparsity(origin, res):
        p_hit = [[],[],[],[],[]]
        user_p = defaultdict(list)
        for user in origin:
            items = list(origin[user].keys())
            for i in range(5):
                predicted = [item[0] for item in res[user][:i+1]]

                user_p[user].append(len(set(items).intersection(set(predicted)))/(i+1))
        return user_p
    
    @staticmethod
    def recall_sparsity(hits, origin):
        user_recall = defaultdict(list)
        for user in hits:
            user_recall[user].append(hits[user]/len(origin[user]))
        return user_recall
    

    @staticmethod
    def NDCG_sparsity(origin,res,N):
        
        user_recall = defaultdict(list)
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            user_recall[user].append(DCG / IDCG)
        return user_recall
    


    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num,5)

    # # @staticmethod
    # def hit_ratio(origin, hits):
    #     """
    #     Note: This type of hit ratio calculates the fraction:
    #      (# users who are recommended items in the test set / #all the users in the test set)
    #     """
    #     hit_num = 0
    #     for user in hits:
    #         if hits[user] > 0:
    #             hit_num += 1
    #     return hit_num / len(origin)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N),5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list),5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return round(error/count,5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return round(math.sqrt(error/count),5)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)

    # @staticmethod
    # def MAP(origin, res, N):
    #     sum_prec = 0
    #     for user in res:
    #         hits = 0
    #         precision = 0
    #         for n, item in enumerate(res[user]):
    #             if item[0] in origin[user]:
    #                 hits += 1
    #                 precision += hits / (n + 1.0)
    #         sum_prec += precision / min(len(origin[user]), N)
    #     return sum_prec / len(res)

    # @staticmethod
    # def AUC(origin, res, rawRes):
    #
    #     from random import choice
    #     sum_AUC = 0
    #     for user in origin:
    #         count = 0
    #         larger = 0
    #         itemList = rawRes[user].keys()
    #         for item in origin[user]:
    #             item2 = choice(itemList)
    #             count += 1
    #             try:
    #                 if rawRes[user][item] > rawRes[user][item2]:
    #                     larger += 1
    #             except KeyError:
    #                 count -= 1
    #         if count:
    #             sum_AUC += float(larger) / count
    #
    #     return float(sum_AUC) / len(origin)


def ranking_evaluation(origin, res, N):
    measure = []

    predicted = {}
    for user in res:
        predicted[user] = res[user][:5]
    p_result = Metric.P(origin,predicted) 
    r_result = []
    for n in [3,5]:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        recall = Metric.recall(hits, origin)
        NDCG = Metric.NDCG(origin, predicted, n)
        r_result.append(recall)
        r_result.append(NDCG)

    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        
        

        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure,p_result,r_result

def ranking_evaluation_for_sparsity(origin, res, N,save_path):
    

    predicted = {}
    for user in res:
        predicted[user] = res[user][:5]
    p_result = Metric.P_sparsity(origin,predicted)

    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
    
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        
        

        hits = Metric.hits(origin, predicted)
        
        recall = Metric.recall_sparsity(hits, origin)

        for user in p_result.keys():
            p_result[user].append(recall[user])
        #indicators.append('Recall:' + str(recall) + '\n')
        
        NDCG = Metric.NDCG_sparsity(origin, predicted, n)

        for user in p_result.keys():
            p_result[user].append(NDCG[user])
    
    
    """ for user in res:
        p_result[user].append(res[user][:5]) """

    print(save_path + "per location has saved")
    with open(save_path, 'wb') as f:
        pickle.dump(p_result, f)
    

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure
