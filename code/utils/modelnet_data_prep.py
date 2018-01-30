import os
import sys
import glob
import h5py
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import threading
from tqdm import tqdm

SAMPLING_BIN = "/home/lqyu/workspace/pcl-pcl-1.8.1/build/bin/pcl_mesh_sampling"
SAMPLING_POINT_NUM = 8192
SAMPLING_LEAF_SIZE = 0.02
SAVE_ROOT_PATH = '../../data/Patches'


def nonuniform_sampling(num = 4096, sample_num = 1024):
    sample = set()
    loc = np.random.rand()*0.6+0.2
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.4)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)

    return list(sample)

def save_file(file_path, data):
    if not os.path.exists(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])
    np.savetxt(file_path, data, fmt='%.6f')

def tmp(file_path=None):
    file_list = glob.glob('/home/lqyu/workspace/PointSR/data/ModelNet10_test/ModelNet10_MC_2k/*.xyz')
    file_list.sort()
    for file_path in file_list:
        print file_path
        data_4096 = np.loadtxt(file_path)
        data = data_4096[:, 0:3]
        centroid = np.mean(data, axis=0, keepdims=True)
        data = data - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(abs(data) ** 2, axis=-1)))
        data = data / furthest_distance
        data_4096[:, 0:3] = data
        save_path = file_path.replace('ModelNet10_MC_2k','ModelNet10_MC_2k_normalize')
        save_file(save_path,data_4096)

        off_path = file_path.replace('ModelNet10_MC_2k','mesh')
        off_path = off_path.replace('.xyz', '.off')
        save_off_path = off_path.replace('mesh', 'mesh_normalize')
        if not os.path.exists(os.path.split(save_off_path)[0]):
            os.makedirs(os.path.split(save_off_path)[0])

        offFile = open(off_path, 'r')
        lines = offFile.readlines()
        offFile.close()
        with open(save_off_path, 'w') as f:
            f.writelines(lines[0:2])
            params = lines[1].split(' ')
            nVert = int(params[0])
            for i in range(2, nVert + 2):
                coord = lines[i].split(' ')
                coords = []
                for item in coord:
                    if item!='':
                        coords.append(item)
                x = (float(coords[0]) - centroid[0, 0]) / furthest_distance
                y = (float(coords[1]) - centroid[0, 1]) / furthest_distance
                z = (float(coords[2]) - centroid[0, 2]) / furthest_distance
                f.write('%.6f %.6f %.6f\n' % (x, y, z))
            f.writelines(lines[nVert + 2:])
        # continue

        # path0 = file_path.replace('poisson_20k','20000_normalize')
        # save_file(path0, data_4096)

        # idx = np.argsort(data_4096[:,np.random.randint(0,2)])
        # data_4096 = data_4096[idx]
        # path1 = file_path.replace('poisson_20k','poisson_5k2')
        # idx1 = nonuniform_sampling(num=len(data_4096), sample_num=len(data_4096)/4)
        # data1 = data_4096[idx1, ...]
        # save_file(path1,data1)


def nonuniformsample_from_pointcloud_fn():
    file_list1 = glob.glob('/home/lqyu/server/proj49/PointSR_data/test_data/our_collected_data/Poisson_20k/*.xyz')
    file_list1.sort()
    # #handle with file_list2 and select the complete whole
    # tmp_list1  = []
    # tmp_list2 = []
    # for item in file_list2:
    #     tmp_item = item.replace('big_girl','biggirl')
    #     if len(tmp_item.split('/')[-1].split('_'))==3:#biggirl_01_1
    #         tmp_list1.append(item)
    #     else:#biggirl_1
    #         tmp_list2.append(item)
    file_list = file_list1

    for file_path in tqdm(file_list):
        print file_path
        phase = file_path.split('/')[-2]
        name = file_path.split('/')[-1].replace("xyz", "xyz")

        data_4096 = np.loadtxt(file_path)
        # data = data_4096[:, 0:3]
        # centroid = np.mean(data, axis=0, keepdims=True)
        # data = data - centroid
        # furthest_distance = np.amax(np.sqrt(np.sum(abs(data) ** 2, axis=-1)))
        # data = data / furthest_distance
        # data_4096[:, 0:3] = data

        path_4096 = file_path
        path_nonuniform_1024 = path_4096.replace('Poisson_20k','Poisson_5k_nonuniform')

        #idx_nonuniform_2048 = nonuniform_sampling(num=4096, sample_num=2048)
        #data_nonuniform_2048 = data_4096[idx_nonuniform_2048, ...]
        sort_idx = np.argsort(data_4096[:, np.random.randint(0, 2)])
        perm_idx = np.random.permutation(np.arange(len(data_4096)))

        idx_nonuniform_1024 = nonuniform_sampling(num=len(data_4096), sample_num=5000)
        data_nonuniform_1024 = data_4096[sort_idx[idx_nonuniform_1024],...]

        # data_nonuniform_1024 = data_4096[perm_idx[:5000], ...]

        #save_file(path_4096,data_4096)
        #save_file(path_nonuniform_2048,data_nonuniform_2048)
        save_file(path_nonuniform_1024,data_nonuniform_1024)


def possion_sample_fn(file_path):
    phase = file_path.split('/')[-2]
    name = file_path.split('/')[-1].replace("off", "xyz")
    xyz_name = os.path.join(SAVE_ROOT_PATH, '20000~', phase, name)
    if not os.path.exists(xyz_name):
        sample_cmd = '../../third_party/PdSampling_nofix %s %s %s' % (str(20000), file_path, xyz_name)
        print sample_cmd
        if os.system(sample_cmd):
            print "cannot sample file: %s" % (file_path)
            return 1

    # xyz_name = os.path.join(SAVE_ROOT_PATH, '4096', phase, name)
    # if not os.path.exists(xyz_name):
    #     sample_cmd = '../../third_party_pc901/PdSampling %s %s %s' % (str(4096), file_path, xyz_name)
    #     print sample_cmd
    #     if os.system(sample_cmd):
    #         print "cannot sample file: %s" % (file_path)
    #         return 1
    #
    # xyz_name = os.path.join(SAVE_ROOT_PATH, '2048', phase, name)
    # if not os.path.exists(xyz_name):
    #     sample_cmd = '../../third_party_pc901/PdSampling %s %s %s' % (str(2048), file_path, xyz_name)
    #     print sample_cmd
    #     if os.system(sample_cmd):
    #         print "cannot sample file: %s" % (file_path)
    #         return 1

    # xyz_name = os.path.join(SAVE_ROOT_PATH, '8192~', phase, name)
    # if os.path.exists(xyz_name):
    #     return
    # print file_path
    # sample_cmd = '../../third_party/PdSampling_nofix %s %s %s' % (str(8192), file_path, xyz_name)
    # if os.system(sample_cmd):
    #     print "cannot sample file: %s" % (file_path)
    #     return 1


def normal_off_file(folder='/home/lqyu/workspace/PointSR/data/ModelNet10'):
    file_list = glob.glob(folder+'/*/train/*.off')
    file_list.sort()
    for file_path in file_list:
        phase = file_path.split('/')[-2]
        name = file_path.split('/')[-1][:-4]+'.xyz'
        data_1024 = np.loadtxt(os.path.join('../../data/ModelNet10_poisson', '1024', phase, name))
        data_2048 = np.loadtxt(os.path.join('../../data/ModelNet10_poisson', '2048', phase, name))
        data_4096 = np.loadtxt(os.path.join('../../data/ModelNet10_poisson', '4096', phase, name))

        data_ori = np.concatenate((data_1024, data_2048, data_4096), axis=0)
        data = data_ori[:, 0:3]
        centroid = np.mean(data, axis=0, keepdims=True)
        data = data - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(abs(data) ** 2, axis=-1)))

        save_off_path = file_path.replace('ModelNet10','ModelNet10_normalize')
        if not os.path.exists(os.path.split(save_off_path)[0]):
            os.makedirs(os.path.split(save_off_path)[0])

        offFile = open(file_path, 'r')
        lines = offFile.readlines()
        offFile.close()
        with open(save_off_path,'w') as f:
            f.writelines(lines[0:2])
            params = lines[1].split(' ')
            nVert = int(params[0])
            for i in range(2,nVert+2):
                coord = lines[i].split(' ')
                x = (float(coord[0])-centroid[0,0])/furthest_distance
                y = (float(coord[1])-centroid[0,1])/furthest_distance
                z = (float(coord[2])-centroid[0,2])/furthest_distance
                f.write('%.6f %.6f %.6f\n'%(x,y,z))
            f.writelines(lines[nVert+2:])


def recalculateNormal_fn(file_path):
    SAVE_ROOT_PATH='../../data/ModelNet10_poisson_normal'
    phase = file_path.split('/')[-2]
    name = file_path.split('/')[-1]
    xyz_name = file_path.split('/')[-1][:-4] + "_p.xyz"
    xyz_normal_name = file_path.split('/')[-1].replace("off", "xyz")

    if os.path.exists(os.path.join(SAVE_ROOT_PATH, '2048_nonuniform', phase, name)):
        return
    data_1024 = np.loadtxt(os.path.join('../../ModelNet10', '1024', phase, name))
    data_2048 = np.loadtxt(os.path.join('../../ModelNet10', '2048', phase, name))
    data_4096 = np.loadtxt(os.path.join('../../ModelNet10', '4096', phase, name))
    data_8196 = np.loadtxt(os.path.join('../../ModelNet10', '20000~', phase, name))

    data = np.concatenate((data_1024, data_2048, data_4096,data_8196), axis=0)
    np.savetxt(xyz_name, data)
    normal_cmd = 'meshlabserver -i %s -o %s -s ../../third_party/calculate_normal.mlx -om vn' % (
        xyz_name, xyz_normal_name)
    if os.system(normal_cmd):
        print "cannot calculate normal file: %s" % (file_path)
        return 1

    data_ori = np.loadtxt(xyz_normal_name)
    data_ori = data_ori[0:1024*7,:]

    data = data_ori[:, 0:3]
    centroid = np.mean(data, axis=0, keepdims=True)
    data = data - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(abs(data) ** 2, axis=-1)))
    data = data / furthest_distance
    data_ori[:, 0:3] = data

    data_1024 = data_ori[0:1024 * 1, :]
    data_2048 = data_ori[1024 * 1:1024 * 3, :]
    data_4096 = data_ori[1024 * 3:1024 * 7, :]

    #generate nonuniform data
    idx_nonuniform_2048 = nonuniform_sampling(num=len(data_4096), sample_num=2048)
    data_nonuniform_2048 = data_4096[idx_nonuniform_2048, ...]
    idx_nonuniform_1024 = nonuniform_sampling(num=len(data_4096), sample_num=1024)
    data_nonuniform_1024 = data_4096[idx_nonuniform_1024, ...]

    # save
    path_1024 = os.path.join(SAVE_ROOT_PATH, '1024', phase, name)
    path_2048 = os.path.join(SAVE_ROOT_PATH, '2048', phase, name)
    path_4096 = os.path.join(SAVE_ROOT_PATH, '4096', phase, name)
    path_nonuniform_1024 = os.path.join(SAVE_ROOT_PATH, '1024_nonuniform', phase, name)
    path_nonuniform_2048 = os.path.join(SAVE_ROOT_PATH, '2048_nonuniform', phase, name)

    if not os.path.exists(os.path.join(SAVE_ROOT_PATH, '1024', phase)):
        os.makedirs(os.path.join(SAVE_ROOT_PATH, '1024', phase))
    if not os.path.exists(os.path.join(SAVE_ROOT_PATH, '2048', phase)):
        os.makedirs(os.path.join(SAVE_ROOT_PATH, '2048', phase))
    if not os.path.exists(os.path.join(SAVE_ROOT_PATH, '4096', phase)):
        os.makedirs(os.path.join(SAVE_ROOT_PATH, '4096', phase))
    if not os.path.exists(os.path.join(SAVE_ROOT_PATH, '1024_nonuniform', phase)):
        os.makedirs(os.path.join(SAVE_ROOT_PATH, '1024_nonuniform', phase))
    if not os.path.exists(os.path.join(SAVE_ROOT_PATH, '2048_nonuniform', phase)):
        os.makedirs(os.path.join(SAVE_ROOT_PATH, '2048_nonuniform', phase))

    np.savetxt(path_1024, data_1024, fmt='%.6f')
    np.savetxt(path_2048, data_2048, fmt='%.6f')
    np.savetxt(path_4096, data_4096, fmt='%.6f')
    np.savetxt(path_nonuniform_1024, data_nonuniform_1024, fmt='%.6f')
    np.savetxt(path_nonuniform_2048, data_nonuniform_2048, fmt='%.6f')

    os.remove(xyz_name)
    os.remove(xyz_normal_name)
    return


def fix_off_file(filepath):
    with open(filepath,'r') as f:
        line = f.readline()
        if line=='OFF\n':
            return
        print filepath
        lines = f.readlines()

    nums = line.split(' ')
    n1 = nums[0][3:]
    n2 = nums[1]
    n3 = nums[2]
    with open(filepath,'w') as f:
        f.write('OFF\n%s %s %s'%(n1,n2,n3))
        f.writelines(lines)


def possion_sample_fn(phase='train'):
    file_list = glob.glob(os.path.join('../../data/ModelNet10', '*', phase, '*.off'))
    file_list.sort()
    new_file_list = []

    for item in file_list:
        name = item.split('/')[-1][:-3]+"xyz"
        xyz_name = os.path.join(SAVE_ROOT_PATH, '20000~', phase,name)
        if not os.path.exists(xyz_name):
            new_file_list.append(item)
    print('Got %d files in modelnet10.' % (len(new_file_list)))
    pool = ThreadPool(8)
    pool.map(possion_sample_fn, new_file_list)

def recalculateNormal(phase='train'):
    file_list = glob.glob(os.path.join('../../ModelNet10/1024',phase,'*.xyz'))
    file_list.sort()
    print('Got %d files in modelnet10.' % (len(file_list)))
    pool = ThreadPool(1)
    pool.map(recalculateNormal_fn, file_list)

def nonuniformsample_from_pointcloud(phase='train'):
    file_list = glob.glob(os.path.join('../../data/surface_with_area/pcl_4096~','*.xyz'))
    file_list.sort()
    print('Got %d files in modelnet10.' % (len(file_list)))
    for item in file_list:
        tmp(item)


def save_h52(save_names = ['poisson_4096','poisson_2048']):
    h5_filename = '/home/lqyu/workspace/PointSR/data/Patches_tt.h5'
    file_names1 = glob.glob(os.path.join('/home/lqyu/server/proj49/PointSR_data/train_data/SHREC',save_names[0],'*.xyz'))
    file_names1.sort()
    #select which data to save
    select_file_names = []
    for name in file_names1:
        id = int(name.split('/')[-1].split('_')[0][1:])
        if id<1000:
            select_file_names.append(name)
    ##read data
    names = []
    catetogy = len(save_names)
    data = [[] for item in range(catetogy)]

    # for item in tqdm(select_file_names):
    #     path1 = os.path.join('/home/lqyu/server/proj49/PointSR_data/train_data/poisson2_4096', item.split('/')[-1])
    #     path2 = os.path.join('/home/lqyu/server/proj49/PointSR_data/train_data/poisson_1024', item.split('/')[-1])
    #     path3 = os.path.join('/home/lqyu/server/proj49/PointSR_data/train_data/MC_4096', item.split('/')[-1])
    #
    #     if os.path.exists(item) and os.path.exists(path1) and os.path.exists(path2):
    #         tmp_data = np.loadtxt(item)
    #         data[0].append(tmp_data)
    #
    #         tmp_data = np.loadtxt(path1)
    #         data[1].append(tmp_data)
    #
    #         tmp_data = np.loadtxt(path2)
    #         data[2].append(tmp_data)
    #
    #         tmp_data = np.loadtxt(path3)
    #         data[3].append(tmp_data)
    #         names.append(item)
    #     else:
    #         print item

    for item in tqdm(select_file_names):

        item_data = []
        for i in range(catetogy):
            path = item.replace(save_names[0],save_names[i])
            tmp_data = np.loadtxt(path)

            centroid = np.mean(tmp_data[:, 0:3], axis=0, keepdims=True)
            tmp_data[:, 0:3] = tmp_data[:, 0:3] - centroid
            furthest_distance = np.amax(np.sqrt(np.sum(abs(tmp_data[:, 0:3]) ** 2, axis=-1)))
            tmp_data[:, 0:3] = tmp_data[:, 0:3] / furthest_distance
            item_data.append(tmp_data)
        if len(item_data)==catetogy:
            names.append(item)
            for i in range(catetogy):
                data[i].append(item_data[i])

    for i in range(catetogy-1):
        assert len(data[i])==len(data[i+1])
    assert len(data[i])==len(names)
    print len(names)
    h5_fout = h5py.File(h5_filename)
    for i in range(catetogy):
        h5_fout.create_dataset(
            save_names[i], data=data[i],
            compression='gzip', compression_opts=4,
            dtype=np.float32)
    string_dt = h5py.special_dtype(vlen=str)
    h5_fout.create_dataset(
        'name', data=names,
        compression='gzip', compression_opts=1,
        dtype=string_dt)
    h5_fout.close()


def save_h5(h5_filename,save_names = ['patch_poisson_4096','patch_poisson_1024_nonuniform']):

    file_names = os.listdir(os.path.join('',save_names[0]))
    file_names.sort()

    #select which data to save
    test_name = ['nicolo','vaselion','bunny','gril']
    select_file_names = []
    for name in file_names:
        mark = False
        for tt in test_name:
            if tt in name:
                mark = True
                break
        if mark==False:
            select_file_names.append(name)

    ##read data
    names = []
    catetogy = len(save_names)
    data = [[] for item in range(catetogy)]
    for item in tqdm(select_file_names):
        item_data = []
        for i in range(catetogy):
            path = os.path.join(pointcloud_path,save_names[i],item)
            tmp_data = np.loadtxt(path)
            if tmp_data.shape[0]==int(save_names[i].split('_')[0]):
                item_data.append(tmp_data)
        if len(item_data)==catetogy:
            names.append(item)
            for i in range(catetogy):
                data[i].append(item_data[i])

    for i in range(catetogy-1):
        assert len(data[i])==len(data[i+1])
    assert len(data[i])==len(names)

    h5_fout = h5py.File(h5_filename)
    for i in range(catetogy):
        h5_fout.create_dataset(
            save_names[i], data=data[i],
            compression='gzip', compression_opts=4,
            dtype=np.float32)
    string_dt = h5py.special_dtype(vlen=str)
    h5_fout.create_dataset(
        'name', data=names,
        compression='gzip', compression_opts=1,
        dtype=string_dt)
    h5_fout.close()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data_1024 = f['data_1024'][:]
    data_4096 = f['data_4096'][:]
    name = f['name'][:]
    return (data_1024,data_4096)

if __name__ == '__main__':
    #nonuniform_subsample_from_pointcloud('/home/lqyu/workspace/PointSR/data/ModelNet10_pc',sampling_num=2048)

    pointcloud_path = '/home/lqyu/workspace/PointSR/data/Patches'
    h5_filename = '/home/lqyu/workspace/PointSR/data/Patches_tt.h5'
    #nonuniformsample_from_pointcloud_fn('aa')
    #nonuniformsample_from_pointcloud_fn('ss')
    #nonuniformsample_from_pointcloud('train')
    #calculateNormal('train')
    #load_h5('/home/lqyu/workspace/PointSR/data/ModelNet10_tt.h5')
    save_h52()
    #normal_off_file()