import numpy as np
import os
import glob


with open("test.txt","w") as f:
    f.write("这是个测试！")  # 自带文件关闭功能，不需要再写f.close()
'''
FOR UCF CRIME
'''
root_path = 'D:\\Code\\Course\\DeepMIL_I3D\\Test\\RGB'
dirs = os.listdir(root_path)
print(dirs)
with open('ucf-i3d-test.list', 'w+') as f:

    normal = []
    for dir in dirs:
        files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
        for file in files:
            if 'x264.' in file:  ## comments
                if 'Normal_' in file:
                    normal.append(file)
                else:
                    newline = file+'\n'
                    f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)
