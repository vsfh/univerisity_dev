import numpy as np

def search_img():
    import random
    from glob import glob
    import numpy as np
    def calcu_cos(feature_query, feature_value):
        print(feature_query, feature_value)
        return np.dot(feature_query, feature_value) / (np.linalg.norm(feature_query) * np.linalg.norm(feature_value))
    search_img_num = 10
    img_list = glob('../../data/features_dict/*_clip.npz')
    res = {}
    for img_path in img_list[:-1]:
        img_name = img_path.split('\\')[-1].split('_')[0]+'_clip.npy'
        query = np.load(f'../../data/features_query/{img_name}', allow_pickle=True)
        sim_list = []
        name_list = []
        ex_img_list = random.sample(img_list, search_img_num-1)
        if img_path in ex_img_list:
            ex_img_list.remove(img_path).append(img_list[-1])
        ex_img_list.append(img_path)

        for npz_path in ex_img_list:
            feature_dict = np.load(npz_path, allow_pickle=True)['arr_0'].item()
            sim_value = [calcu_cos(feature, query) for _, feature in feature_dict.items()]
            sim_list.append(sim_value)
            name_list.append(npz_path.split('\\')[-1].split('_')[0])
        rank = np.argsort(-np.array(sim_list))[-1]
        print('rank: ', rank)
        res[img_name] = {'name': name_list, 'sim': sim_list}
        break
    np.savez('search_res.npz', res)

def rank(name):
    res = np.load(f'search_res_{name}.npz', allow_pickle=True)['arr_0'].item()
    rank_list = []
    for file_name, value in res.items():
        name_list = value['name']
        sim_list = np.array(value['sim'])
        top2_max = np.partition(sim_list, -2, axis=1)[:, -2:]
        row_avg = np.mean(top2_max, axis=1).reshape(-1, 1)
        rank_list.append(sum(row_avg>row_avg[-1])+1)
    rank_arr = np.array(rank_list)
    print(name)
    print('rank1: ', sum(rank_arr==1))
    print('rank1: ', sum(rank_arr<=2))
    print('rank5: ', sum(rank_arr<=5))
    print('rank10: ', sum(rank_arr<=10))

    pass

def align():
    import random
    from tqdm import tqdm
    def calcu_cos(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm[...,None]).flatten()
    res = {}
    top1 = 0
    top5 = 0
    top10 = 0
    feature1 = np.load('/home/SATA4T/gregory/data/dino_drone_feature.npz')
    # feature2 = np.load('dino_features_dict.npz')
    feature2 = np.load('/home/SATA4T/gregory/data/dino_finetune_feature.npz')
    img_list = [name for name in feature2.keys()]
    for img_name in tqdm(img_list):
        if img_name == 'test':
            continue
        drone_feature = feature1[img_name]
        ex_img_list = random.sample(img_list, 99)
        if img_name in ex_img_list:
            ex_img_list.remove(img_name)
            ex_img_list.append(img_list[-1])
        ex_img_list.append(img_name)
        res[img_name] = []
        for ex_name in ex_img_list:
            res[img_name].append(calcu_cos(drone_feature, feature2[ex_name]))
        img_res = np.array(res[img_name]).mean(1).argsort()[-15:][::-1]
        if 99 in img_res[:1]:
            top1 += 1
        if 99 in img_res[:5]:
            top5 += 1
        if 99 in img_res[:10]:
            top10 += 1

    np.savez('res.npz', **res)
    print(top1, top5, top10)

align()
# rank('clip_l')
# rank('clip')
# rank('eva_clip_l')
# rank('eva_clip_b')
# rank('siglip_l')
# rank('siglip')