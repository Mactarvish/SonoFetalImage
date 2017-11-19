import os
import shutil
import MaUtilities as mu
from PIL import Image
mu.image_path = '/home/hdl2/Desktop/Sono_nofolders/%s.jpg'
mu.save_path = None

'''
Preprocess for those images.
'''


root = "/home/hdl2/Desktop/SonoScape201705/"
#root = "/home/hdl2/Desktop/Test Folder/"



# delete second level folders
def delete_second_folders():
    n_900226 = 0
    for sub_folder in os.listdir(root):
        if "900226" in sub_folder:
            n_900226 += 1
        else:
            print(sub_folder)
    print("Total %d %d folders" % (len(os.listdir(root)), n_900226))
    n_folders = len(os.listdir(root))
    n_moved = 0
    for sub_folder in os.listdir(root):
        print(sub_folder)
        for sub_sub_folder in os.listdir(root + sub_folder):
            print(sub_sub_folder)
            for file in os.listdir(root + sub_folder + '/' + sub_sub_folder):
                print(file)
                os.rename(root + sub_folder + '/' + sub_sub_folder + '/' + file, root + sub_folder + '/' + file)
            os.rmdir(root + sub_folder + '/' + sub_sub_folder)
            n_moved += 1
            print("Process: %d/%d" %(n_moved, n_folders))

# Has deleted second level folders, reset file name to subfolder name added by index
def copy_all_files():
    n_files = 0
    n_copied_files = 0
    for sub_folder in os.listdir(root):
        print(sub_folder)
        n_files += len(os.listdir(root + sub_folder))
        image_index = 0
        video_index = 0
        for file in os.listdir(root + sub_folder):
            #print(file)
            new_file_name = file
            if "jpg" in file:
                new_file_name = sub_folder + "_%d" % image_index + ".jpg"
                image_index += 1
            elif "wmv" in file:
                new_file_name = sub_folder + "_%d" % video_index + ".wmv"
                video_index += 1
            print(new_file_name)
            shutil.copyfile(root + sub_folder + '/' + file, "/home/hdl2/Desktop/Sono_nofolder/" + new_file_name)
            n_copied_files += 1
            print('Process: %d/14449' % n_copied_files)

    #print(n_files)
    #print(len(os.listdir(root)))

# No folders now, copy all images to a new folder and rename them index
def copy_all_images():
    n_files = len(os.listdir(root))
    n_copied_files = 0
    for file in os.listdir(root):
        if ".jpg" in file:
            image = file
            new_image_name = str(n_copied_files) + ".jpg"
            shutil.copyfile(root + image, "/home/hdl2/Desktop/SonoDataset/" + new_image_name)
            n_copied_files += 1
            print('Process: %d/%d' % (n_copied_files, n_files))

# resize current images to small size for classify.
def resize_images():
    for index, file_name in enumerate(os.listdir(root)):
        if "git" not in file_name:
            image_name = file_name
            image = Image.open(root + image_name)
            image = image.resize((224, 224))
            image.save(root + image_name)
            print(index)

def remove_900226():
    for sub_folder in os.listdir(root):
        print(sub_folder)
        if "900226" in sub_folder:
            pos = sub_folder.find("900226")
            new_name = ''
            if sub_folder[pos+6] == '_':
                new_name = sub_folder.replace("900226_", '')
            else:
                new_name = sub_folder.replace("900226", '')
            if new_name[-1] == '_':
                new_name = new_name[:-1]
            os.rename(root + sub_folder, root + new_name)

def remove_last_():
    for sub_folder in os.listdir(root):
        if sub_folder[-1] == '_':
            new_name = sub_folder[:-1]
            os.rename(root + sub_folder, root + new_name)



def set_label_mark(file):
    if ".jpg" in file:
        file = file[:-4]
    image_path = "/home/hdl2/Desktop/Sono_nofolder/%s.jpg"
    a = mu.crop_image(image_path % file, (128, 1148), (80, 200), mode=None)
    #mu.display(image_path, a)
    n_yellow= 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            pixel_color = a[i, j, :]
            if ((pixel_color[0] - 255) ** 2 + (pixel_color[1] - 255) ** 2 + (pixel_color[2]) ** 2) < 20:
                n_yellow += 1
    if n_yellow > 50:
        os.rename(image_path % file, image_path % (file + "_Labeled"))
        print(file, "Labeled")
    else:
        os.rename(image_path % file, image_path % (file + "_No"))
        print(file, "NO")


def get_circle_image_set():
    for file in os.listdir("/home/hdl2/Desktop/Sono_nofolder/"):
        print(file)
        if "_Circle" in file:
            shutil.copyfile("/home/hdl2/Desktop/Sono_nofolder/" + file, "/home/hdl2/Desktop/Circle_Images/" + file)

#set_label_mark()

root = "/home/hdl2/Desktop/SonoDataset/Images/"
#copy_all_images()
resize_images()