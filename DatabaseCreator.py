import sqlite3
import shutil
import os
import MaUtilities as mu
import random

def CreateTable():
    connection = sqlite3.connect("ThyDataset")
    cu = connection.cursor()
    cu.execute("CREATE TABLE IF NOT EXISTS Train (id            INTEGER      PRIMARY KEY AUTOINCREMENT,"
                                                 "imagename     VARCHAR(100) UNIQUE,"
                                                 "imagecategory VARCHAR(100) NOT NULL)")


    cu.execute("CREATE TABLE IF NOT EXISTS Validation (id            INTEGER      PRIMARY KEY AUTOINCREMENT,"
                                                "imagename     VARCHAR(100) UNIQUE,"
                                                "imagecategory VARCHAR(100) NOT NULL)")

CreateTable()
test_dir = "/home/hdl2/Desktop/SonoFetalImage/TestFolder/"

def RenameFile(original_dir, original_filename, new_filename):
    shutil.copyfile(original_dir + '/' + original_filename, original_dir + '/' + new_filename)
    os.remove(original_dir + '/' + original_filename)
    print("rename file %s to %s in %s" % (str(original_filename), str(new_filename), str(original_dir)))

#RenameFile(test_dir, 'aaassss', 'aaasssss')

dir_hashimoto_thyroiditis1 = "/home/hdl2/Desktop/MedicalImage/hashimoto_thyroiditis1" # 650
dir_hyperthyreosis1        = "/home/hdl2/Desktop/MedicalImage/hyperthyreosis1"        # 1250
dir_normal1                = "/home/hdl2/Desktop/MedicalImage/normal1"                # 800
dir_postoperative1         = "/home/hdl2/Desktop/MedicalImage/postoperative1"         # 200
dir_subacute_thyroiditis1  = "/home/hdl2/Desktop/MedicalImage/subacute_thyroiditis1"  # 1350
dir_subhyperthyreosis1     = "/home/hdl2/Desktop/MedicalImage/subhyperthyreosis1"     # 200

dirs = [dir_hashimoto_thyroiditis1, dir_hyperthyreosis1, dir_normal1, dir_postoperative1, dir_subacute_thyroiditis1, dir_subhyperthyreosis1]

def AppendFoldernamesToimageNamesInfront():
    for dir in dirs:
        category = dir.split('/')[-1]
        images = os.listdir(dir)
        count = 0
        for image in images:
            RenameFile(dir, image, category + '_' + image)
            count += 1
            print(count, '/', len(images), category + '_' + image)
        # print(images)
        # print(len(images))

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
        # distribute 50 images every category for Validation
        if num_images - count > 100:
            train_set.append((image_name, category))
            # cu.execute("insert into Train (imagename, imagecategory) values ('%s', '%s')" % (image_name, category))
        else:
            validation_set.append((image_name, category))
            # cu.execute("insert into Validation (imagename, imagecategory) values ('%s', '%s')" % (image_name, category))
        count += 1
        print(count, '/', num_images)
    # connection.commit()
random.shuffle(train_set)
random.shuffle(validation_set)
print(train_set)
print(validation_set)
for train_sample in train_set:
    cu.execute("insert into Train (imagename, imagecategory) values ('%s', '%s')" % (train_sample[0], train_sample[1]))
for validation_sample in validation_set:
    cu.execute("insert into Validation (imagename, imagecategory) values ('%s', '%s')" % (validation_sample[0], validation_sample[1]))
connection.commit()
