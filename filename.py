import os
import shutil
import pypinyin
import pandas as pd

if __name__ == '__main__':
    # 定义加载和保存的文件夹
    load_case_dir_path = '/home/user16/sharedata/GXE/SkullWidth/data/AI-cbct测量'
    save_case_dir_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'

    # 开始计数和对应的CSV文件，对应的原先的中文名列表和重命名后新的英文名列表
    start_counter = 1
    save_csv_file_path = '/home/user16/sharedata/GXE/SkullWidth/data/image_data.csv'
    old_case_name_list = []
    new_case_name_list = []

    # 遍历所有的数据案例进行标准化
    for zh_case_name in os.listdir(load_case_dir_path):
        if len(os.listdir(os.path.join(load_case_dir_path, zh_case_name))) == 0:
            print('%s 这个案例下面没有数据，请进行认真核查!' % zh_case_name)
            continue
        # 将中文名字转换为拼音
        en_case_name = ''.join(pypinyin.lazy_pinyin(zh_case_name))
        # 标准化重命名
        old_fold_name = zh_case_name
        new_fold_name = '%03d_%s' % (start_counter, en_case_name)
        start_counter = start_counter + 1
        # 进行数据的复制
        shutil.copytree(os.path.join(load_case_dir_path, old_fold_name), os.path.join(save_case_dir_path, new_fold_name))
        # 对应的变量添加入列表
        old_case_name_list.append(old_fold_name)
        new_case_name_list.append(new_fold_name)

    # 保存对应关系为CSV文件，方便以后进行回顾性查看
    dict_data = {'标准的英文名': new_case_name_list, '原先的中文名': old_case_name_list}
    df = pd.DataFrame(dict_data)
    df.to_csv(save_csv_file_path)
