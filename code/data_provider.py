import numpy as np
import h5py
import time
import Queue
import threading
import cv2
from utils import show3d
import os

def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance
    return input, centroid,furthest_distance

def load_patch_data(h5_filename='../h5_data/Patches_noHole_and_collected.h5', skip_rate = 1,num_point=2048, use_randominput=True, norm=False):
    if use_randominput:
        print "use randominput, input h5 file is:", h5_filename
        f = h5py.File(h5_filename)
        input = f['poisson_4096'][:]
        gt = f['poisson_4096'][:]
    else:
        print "Do not randominput, input h5 file is:",h5_filename
        f = h5py.File(h5_filename)
        gt = f['poisson_4096'][:]
        input = f['montecarlo_1024'][:]

    name = f['name'][:]
    assert len(input) == len(gt)

    if norm:
        print "Normalization the data"
        data_radius = np.ones(shape=(len(input)))
        centroid = np.mean(gt[:,:,0:3], axis=1, keepdims=True)
        gt[:,:,0:3] = gt[:,:,0:3] - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(gt[:,:,0:3] ** 2, axis=-1)),axis=1,keepdims=True)
        gt[:, :, 0:3] = gt[:,:,0:3] / np.expand_dims(furthest_distance,axis=-1)
        input[:, :, 0:3] = input[:, :, 0:3] - centroid
        input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance,axis=-1)
    else:
        print "Do not normalization the data"
        centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((gt[:, :, 0:3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        data_radius = furthest_distance[0,:]

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    name = name[::skip_rate]

    object_name = list(set([item.split('/')[-1].split('_')[0] for item in name]))
    object_name.sort()
    print "load object names {}".format(object_name)
    print "total %d samples" % (len(input))
    return input, gt, data_radius, name


def rotate_point_cloud_and_gt(batch_data,batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        # rotation_angle = np.random.uniform(size=(3)) * 2 * np.pi
        # cosval = np.cos(rotation_angle)
        # sinval = np.sin(rotation_angle)
        # rotation_matrix = np.array([[cosval, 0, sinval],
        #                             [0, 1, 0],
        #                             [-sinval, 0, cosval]])

        batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
        if batch_data.shape[-1]>3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        if batch_gt is not None:
            batch_gt[k, ..., 0:3]   = np.dot(batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt.shape[-1] > 3:
                batch_gt[k, ..., 3:] = np.dot(batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

    return batch_data,batch_gt


def shift_point_cloud_and_gt(batch_data, batch_gt = None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] += shifts[batch_index,0:3]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    return batch_data,batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt = None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] *= scales[batch_index]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] *= scales[batch_index]

    return batch_data,batch_gt,scales


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in xrange(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        batch_data[k, ...,0:3] = np.dot(batch_data[k, ...,0:3].reshape((-1, 3)), R)
        if batch_data.shape[-1]>3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), R)

    return batch_data


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data[:,:,3:] = 0
    jittered_data += batch_data
    return jittered_data


def save_pl(path, pl):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    myfile = file(path, "w")
    point_num = pl.shape[0]
    for j in range(point_num):
        if len(pl[j])==3:
            print >> myfile, "%f %f %f" % (pl[j,0],pl[j,1],pl[j,2])
        elif len(pl[j])==6:
            print >> myfile, "%f %f %f %f %f %f" % (pl[j, 0], pl[j, 1], pl[j, 2],pl[j, 3],pl[j, 4],pl[j, 5])
            # print >> myfile, "%f %f %f %f %f %f %f" % (
            # pl[j, 0], pl[j, 1], pl[j, 2], pl[j, 3], pl[j, 4], pl[j, 5], pl[j, 2])
        elif len(pl[j])==7:
            print >> myfile, "%f %f %f %f %f %f %f" % (pl[j, 0], pl[j, 1], pl[j, 2],pl[j, 3],pl[j, 4],pl[j, 5],pl[j, 2])
    myfile.close()
    if np.random.rand()>1.9:
        show3d.showpoints(pl[:, 0:3])


def nonuniform_sampling(num = 4096, sample_num = 1024):
    sample = set()
    loc = np.random.rand()*0.8+0.1
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.3)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)
    return list(sample)


class Fetcher(threading.Thread):
    def __init__(self, input_data, gt_data, radius_data,batch_size,num_point, use_random_input,use_norm):
        super(Fetcher,self).__init__()
        self.queue = Queue.Queue(50)
        self.stopped = False
        self.input_data = input_data
        self.gt_data = gt_data
        self.radius_data = radius_data
        self.batch_size = batch_size
        self.num_point = num_point
        self.use_random_input = use_random_input
        self.use_norm = use_norm
        self.sample_cnt = self.input_data.shape[0]
        self.num_batches = self.sample_cnt//self.batch_size
        print "NUM_BATCH is %s"%(self.num_batches)
        print self.use_random_input,self.use_norm

    def run(self):
        while not self.stopped:
            idx = np.arange(self.sample_cnt)
            np.random.shuffle(idx)
            self.input_data = self.input_data[idx, ...]
            self.gt_data = self.gt_data[idx, ...]
            self.radius_data = self.radius_data[idx, ...]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()
                batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()
                radius = self.radius_data[start_idx:end_idx].copy()
                if self.use_random_input:
                    new_batch_input = np.zeros((self.batch_size, self.num_point,batch_input_data.shape[2]))
                    for i in xrange(self.batch_size):
                        idx = nonuniform_sampling(self.input_data.shape[1], sample_num=self.num_point)
                        new_batch_input[i, ...] = batch_input_data[i][idx]
                    batch_input_data = new_batch_input
                if self.use_norm:
                    batch_input_data, batch_data_gt = rotate_point_cloud_and_gt(batch_input_data,batch_data_gt)
                    batch_input_data, batch_data_gt, scales = random_scale_point_cloud_and_gt( batch_input_data, batch_data_gt,
                                                                                               scale_low=0.9, scale_high=1.1)
                    radius = radius * scales
                    batch_input_data, batch_data_gt = shift_point_cloud_and_gt(batch_input_data, batch_data_gt,shift_range=0.1)
                    if np.random.rand() > 0.5:
                        if self.use_random_input:
                            batch_input_data = jitter_perturbation_point_cloud(batch_input_data, sigma=0.025,clip=0.05)
                        else:
                            batch_input_data = jitter_perturbation_point_cloud(batch_input_data, sigma=0.025, clip=0.05)
                    if np.random.rand() > 0.5:
                        if self.use_random_input:
                            batch_input_data = rotate_perturbation_point_cloud(batch_input_data, angle_sigma=0.03, angle_clip=0.09)
                        else:
                            batch_input_data = rotate_perturbation_point_cloud(batch_input_data, angle_sigma=0.03, angle_clip=0.09)

                else:
                    batch_input_data, batch_data_gt = rotate_point_cloud_and_gt(batch_input_data,batch_data_gt)
                    batch_input_data, batch_data_gt, scales = random_scale_point_cloud_and_gt(batch_input_data,batch_data_gt,
                                                                                              scale_low=0.9,scale_high=1.5)
                    radius = radius * scales
                    batch_input_data, batch_data_gt = shift_point_cloud_and_gt(batch_input_data,batch_data_gt, shift_range=0.3)

                    batch_input_data = jitter_perturbation_point_cloud(batch_input_data, sigma=0.005,
                                                                                     clip=0.02)
                    batch_input_data = rotate_perturbation_point_cloud(batch_input_data,angle_sigma=0.02, angle_clip=0.04)
                    if np.random.rand() > 0.5:
                        batch_input_data = jitter_perturbation_point_cloud(batch_input_data, sigma=0.005, clip=0.02)
                    if np.random.rand() > 0.5:
                        batch_input_data = rotate_perturbation_point_cloud(batch_input_data, angle_sigma=0.02, angle_clip=0.04)
                self.queue.put((batch_input_data, batch_data_gt,radius))
        return None
    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print "Shutdown ....."
        while not self.queue.empty():
            self.queue.get()
        print "Remove all queue data"

if __name__ == '__main__':
    folder = '/home/lqyu/workspace/PointSR/perfect_models'
    fetchworker = Fetcher(folder)
    fetchworker.start()

    for cnt in xrange(200):
        start = time.time()
        input,gt,radius = fetchworker.fetch()
        assert len(input)==len(gt)
        assert len(input)==32
        end = time.time()
        print cnt,end-start
        for i in range(len(input)):
            cv2.imshow('data',input[i,:,0:3])
            while True:
                cmd = show3d.showpoints(input[i,:,0:3])
                if cmd == ord(' '):
                    break
                elif cmd == ord('q'):
                    break
            if cmd == ord('q'):
                break

    fetchworker.shutdown()
