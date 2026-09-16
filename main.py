from SELFRec import SELFRec
from util.conf import ModelConf
import numpy as np 
import torch
import random
import os
import time
def set_seed(seed: int = 42):
    random.seed(seed)                       # Python 随机模块
    np.random.seed(seed)                    # Numpy 随机模块
    torch.manual_seed(seed)                 # CPU 随机种子
    torch.cuda.manual_seed(seed)            # 当前GPU随机种子
    torch.cuda.manual_seed_all(seed)        # 所有GPU随机种子

    torch.backends.cudnn.deterministic = True   # 确保每次卷积结果一致
    torch.backends.cudnn.benchmark = False      # 禁止cudnn自动优化（否则可能不确定）



if __name__ == '__main__':
    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)
    model = 'MRLMRP'
    seed = 42
    dataset = 'Li'
    k = 3
    coe = 0.3

    set_seed(seed)
    s = time.time()
    conf = ModelConf('./conf/' + model + '.conf')
    conf['training.set'] = conf['training.set'].replace("Li",dataset)
    conf['test.set'] = conf['test.set'].replace("Li",dataset)
    conf['dev.set'] = conf['dev.set'].replace("Li",dataset)
    if conf.contain('mineral_embedding_path'):
        conf['mineral_dict_path'] = conf['mineral_dict_path'].replace("Li",dataset)
    conf['k'] = str(k)
    conf['coe'] = str(coe)
    rec = SELFRec(conf)
    p,r = rec.execute()
    case_result = [p[0],r[0],r[1]]
    
    e = time.time()
    print("model:{} dataset:{} seed:{} k:{} coe:{}".format(model,dataset,seed,k,coe))
    print("Running time: %f s" % (e - s))
    print("procedure over")