import pickle
import numpy as np

def iid_sampling(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


f = open('/home/xwchi/MISA/datasets/MOSEI/train.pkl','rb')
data = pickle.load(f)

user_dict = iid_sampling(data, num_users=30)
for idx in user_dict.keys():
    tmp_data = []
    for data_id in user_dict[idx]: tmp_data.append(data[data_id])
    with open(f'/home/xwchi/MISA/datasets/MOSEI/train_{idx}.pkl','wb') as f:
        pickle.dump(tmp_data, f)

exit(0)