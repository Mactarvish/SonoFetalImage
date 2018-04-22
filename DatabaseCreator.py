import sqlite3
import shutil
import os
import MaUtilities as mu
import random

def CreateImageDatabase(database_name):
    '''
    创建名为database_name的sqlite3数据库文件，由两张空表：Train和Validation组成。
    每一张表有三个指标：id，imagename和imagecategory，分别对应图像在该文件中的编号，图像文件的名称（包含后缀名），以及图像的类别。
    :return: 
    '''
    # 如果database_name已经存在，则抛出异常
    assert not os.path.exists(database_name), "database %s already exists" % database_name
    # 如果database_name不存在，则会自动创建
    connection = sqlite3.connect(database_name)
    cu = connection.cursor()
    cu.execute("CREATE TABLE IF NOT EXISTS Train (id            INTEGER      PRIMARY KEY AUTOINCREMENT,"
                                                 "imagename     VARCHAR(100) UNIQUE,"
                                                 "imagecategory VARCHAR(100) NOT NULL)")


    cu.execute("CREATE TABLE IF NOT EXISTS Validation (id            INTEGER      PRIMARY KEY AUTOINCREMENT,"
                                                "imagename     VARCHAR(100) UNIQUE,"
                                                "imagecategory VARCHAR(100) NOT NULL)")

dir_hashimoto_thyroiditis1 = "/home/hdl2/Desktop/MedicalImage/hashimoto_thyroiditis1" # 650
dir_hyperthyreosis1        = "/home/hdl2/Desktop/MedicalImage/hyperthyreosis1"        # 1250
dir_normal1                = "/home/hdl2/Desktop/MedicalImage/normal1"                # 800
dir_postoperative1         = "/home/hdl2/Desktop/MedicalImage/postoperative1"         # 200
dir_subacute_thyroiditis1  = "/home/hdl2/Desktop/MedicalImage/subacute_thyroiditis1"  # 1350
dir_subhyperthyreosis1     = "/home/hdl2/Desktop/MedicalImage/subhyperthyreosis1"     # 200

dirs = [dir_hashimoto_thyroiditis1, dir_hyperthyreosis1, dir_normal1, dir_postoperative1, dir_subacute_thyroiditis1, dir_subhyperthyreosis1]
#dirs = [dir_hashimoto_thyroiditis1, dir_hyperthyreosis1, dir_normal1, dir_subacute_thyroiditis1]

def AppendFoldernamesToimageNamesInfront(dirs):
    '''
    :param dirs: 每一类图像所在的文件夹路径的集合， 例如 ["/home/hdl2/Desktop/MedicalImage/subhyperthyreosis1", "/home/hdl2/Desktop/MedicalImage/hyperthyreosis1"]
    :return: 
    '''
    for dir in dirs:
        category = dir.split('/')[-1]
        images = os.listdir(dir)
        count = 0
        for image in images:
            mu.RenameFile(dir, image, category + '_' + image)
            count += 1
            print(count, '/', len(images), category + '_' + image)
        # print(images)
        # print(len(images))

# connection = sqlite3.connect("ThyDataset")
# cu = connection.cursor()
# train_set = []
# validation_set = []
# for dir in dirs:
#     category = dir.split('/')[-1]
#     image_names = os.listdir(dir)
#     # random.shuffle(image_names)
#     num_images = len(image_names)
#     count = 0
#     for image_name in image_names:
#         # distribute 1/5 images every category for Validation
#         if num_images - count > num_images/5:
#             train_set.append((image_name, category))
#             # cu.execute("insert into Train (imagename, imagecategory) values ('%s', '%s')" % (image_name, category))
#         else:
#             validation_set.append((image_name, category))
#             # cu.execute("insert into Validation (imagename, imagecategory) values ('%s', '%s')" % (image_name, category))
#         count += 1
#         print(count, '/', num_images)
#     # connection.commit()
# random.shuffle(train_set)
# random.shuffle(validation_set)
# print(train_set)
# print(validation_set)
# for train_sample in train_set:
#     cu.execute("insert into Train (imagename, imagecategory) values ('%s', '%s')" % (train_sample[0], train_sample[1]))
# for validation_sample in validation_set:
#     cu.execute("insert into Validation (imagename, imagecategory) values ('%s', '%s')" % (validation_sample[0], validation_sample[1]))
# connection.commit()

def CreateThyDataset(dirs, tv_proportion=5, shuffle=True):
    '''
    向创建好的空数据库文件中写入条目，创建甲状腺PET图像数据集
    :param tv_proportion: 训练集和验证集（测试集）大小的比值
    :param shuffle: 是否打乱每条数据
    :param dirs: 每一类图像所在的文件夹路径的集合， 例如 ["/home/hdl2/Desktop/MedicalImage/subhyperthyreosis1", "/home/hdl2/Desktop/MedicalImage/hyperthyreosis1"]
    
    :return: 
    '''
    connection = sqlite3.connect("ThyDataset")
    cu = connection.cursor()
    train_set = []
    validation_set = []
    for dir in dirs:
        category = dir.split('/')[-1]
        image_names = os.listdir(dir)
        # random.shuffle(image_names)
        num_images = len(image_names)
        count = 0
        for image_name in image_names:
            # distribute 1/5 images every category for Validation
            if num_images - count > num_images / tv_proportion:
                train_set.append((image_name, category))
                # cu.execute("insert into Train (imagename, imagecategory) values ('%s', '%s')" % (image_name, category))
            else:
                validation_set.append((image_name, category))
                # cu.execute("insert into Validation (imagename, imagecategory) values ('%s', '%s')" % (image_name, category))
            count += 1
            print(count, '/', num_images)
            # connection.commit()
    if shuffle:
        random.shuffle(train_set)
        random.shuffle(validation_set)
    print(train_set)
    print(validation_set)
    for train_sample in train_set:
        cu.execute(
            "insert into Train (imagename, imagecategory) values ('%s', '%s')" % (train_sample[0], train_sample[1]))
    for validation_sample in validation_set:
        cu.execute("insert into Validation (imagename, imagecategory) values ('%s', '%s')" % (
        validation_sample[0], validation_sample[1]))
    connection.commit()

CreateImageDatabase("ThyDataset")
CreateThyDataset(dirs, tv_proportion=5, shuffle=True)