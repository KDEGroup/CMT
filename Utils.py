import torch
import random
import numpy as np
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class Logging():
    def __init__(self, filename):
        self.filename = filename

    def record(self, str_log):
        filename = self.filename
        print(str_log)
        with open(filename, 'a') as f:
            f.write("%s\r" % str_log)
            f.flush()