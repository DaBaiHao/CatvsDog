#%%
import os

train = 'train'

dogs = [train+'/'+i for i in os.listdir(train) if 'dog' in i]
cats = [train+'/'+i for i in os.listdir(train) if 'cat' in i]

print(len(dogs), len(cats))

#%%
import os

def createDir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print("创建文件夹失败")
            exit(1)


path = 'min_trainfordata/'

createDir(path+'train/dogs')
createDir(path+'train/cats')
createDir(path+'test/dogs')
createDir(path+'test/cats')

import shutil

for dog, cat in list(zip(dogs, cats))[:1000]:
    shutil.copyfile(dog, path+'train/dogs/'+os.path.basename(dog))
    print(os.path.basename(dog) + '操作成功')
    shutil.copyfile(cat, path+'train/cats/'+os.path.basename(cat))
    print(os.path.basename(cat) + '操作成功')
for dog, cat in list(zip(dogs, cats))[1000:1500]:
    shutil.copyfile(dog, path+'test/dogs/'+os.path.basename(dog))
    print(os.path.basename(dog) + '操作成功')
    shutil.copyfile(cat, path + 'test/cats/'+os.path.basename(cat))
    print(os.path.basename(cat) + '操作成功')

