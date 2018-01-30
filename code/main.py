import argparse
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import socket
from matplotlib import pyplot as plt
import model_generator2_2new6 as MODEL_GEN
import model_utils
import data_provider
from utils import pc_util

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', help='train or test [default: train]')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../model/generator2_new6', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4,   help='Upsampling Ratio [default: 2]')
parser.add_argument('--max_epoch', type=int, default=120, help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=28, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001)

ASSIGN_MODEL_PATH=None
USE_DATA_NORM = True
USE_RANDOM_INPUT = True
USE_REPULSION_LOSS = True

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
UP_RATIO = FLAGS.up_ratio
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MODEL_DIR = FLAGS.log_dir

print socket.gethostname()
print FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

def train(assign_model_path=None):
    is_training = True
    bn_decay = 0.95
    step = tf.Variable(0,trainable=False)
    learning_rate = BASE_LEARNING_RATE
    tf.summary.scalar('bn_decay', bn_decay)
    tf.summary.scalar('learning_rate', learning_rate)

    # get placeholder
    pointclouds_pl, pointclouds_gt, pointclouds_gt_normal, pointclouds_radius = MODEL_GEN.placeholder_inputs(BATCH_SIZE, NUM_POINT, UP_RATIO)

    #create the generator model
    pred,_ = MODEL_GEN.get_gen_model(pointclouds_pl, is_training, scope='generator',bradius=pointclouds_radius,
                                                          reuse=None,use_normal=False, use_bn=False,use_ibn=False,
                                                          bn_decay=bn_decay,up_ratio=UP_RATIO)

    #get emd loss
    gen_loss_emd,matchl_out = model_utils.get_emd_loss(pred, pointclouds_gt, pointclouds_radius)

    #get repulsion loss
    if USE_REPULSION_LOSS:
        gen_repulsion_loss = model_utils.get_repulsion_loss4(pred)
        tf.summary.scalar('loss/gen_repulsion_loss', gen_repulsion_loss)
    else:
        gen_repulsion_loss =0.0

    #get total loss function
    pre_gen_loss = 100 * gen_loss_emd + gen_repulsion_loss + tf.losses.get_regularization_loss()

    # create pre-generator ops
    gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

    with tf.control_dependencies(gen_update_ops):
        pre_gen_train = tf.train.AdamOptimizer(learning_rate,beta1=0.9).minimize(pre_gen_loss,var_list=gen_tvars,
                                                                                 colocate_gradients_with_ops=True,
                                                                                 global_step=step)
    # merge summary and add pointclouds summary
    tf.summary.scalar('loss/gen_emd', gen_loss_emd)
    tf.summary.scalar('loss/regularation', tf.losses.get_regularization_loss())
    tf.summary.scalar('loss/pre_gen_total', pre_gen_loss)
    pretrain_merged = tf.summary.merge_all()

    pointclouds_image_input = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
    pointclouds_input_summary = tf.summary.image('pointcloud_input', pointclouds_image_input, max_outputs=1)
    pointclouds_image_pred = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
    pointclouds_pred_summary = tf.summary.image('pointcloud_pred', pointclouds_image_pred, max_outputs=1)
    pointclouds_image_gt = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
    pointclouds_gt_summary = tf.summary.image('pointcloud_gt', pointclouds_image_gt, max_outputs=1)
    image_merged = tf.summary.merge([pointclouds_input_summary,pointclouds_pred_summary,pointclouds_gt_summary])

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, 'train'), sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_gt': pointclouds_gt,
               'pointclouds_gt_normal':pointclouds_gt_normal,
               'pointclouds_radius': pointclouds_radius,
               'pointclouds_image_input':pointclouds_image_input,
               'pointclouds_image_pred': pointclouds_image_pred,
               'pointclouds_image_gt': pointclouds_image_gt,
               'pretrain_merged':pretrain_merged,
               'image_merged': image_merged,
               'gen_loss_emd': gen_loss_emd,
               'pre_gen_train':pre_gen_train,
               'pred': pred,
               'step': step,
               }
        #restore the model
        saver = tf.train.Saver(max_to_keep=6)
        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(MODEL_DIR)
        global LOG_FOUT
        if restore_epoch==0:
            LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'w')
            LOG_FOUT.write(str(socket.gethostname()) + '\n')
            LOG_FOUT.write(str(FLAGS) + '\n')
        else:
            LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'a')
            saver.restore(sess,checkpoint_path)

        ###assign the generator with another model file
        if assign_model_path is not None:
            print "Load pre-train model from %s"%(assign_model_path)
            assign_saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith("generator")])
            assign_saver.restore(sess, assign_model_path)

        ##read data
        input_data, gt_data, data_radius, _ = data_provider.load_patch_data(skip_rate=1, num_point=NUM_POINT, norm=USE_DATA_NORM,
                                                                                              use_randominput = USE_RANDOM_INPUT)

        fetchworker = data_provider.Fetcher(input_data,gt_data,data_radius,BATCH_SIZE,NUM_POINT,USE_RANDOM_INPUT,USE_DATA_NORM)
        fetchworker.start()
        for epoch in tqdm(range(restore_epoch,MAX_EPOCH+1),ncols=55):
            log_string('**** EPOCH %03d ****\t' % (epoch))
            train_one_epoch(sess, ops, fetchworker, train_writer)
            if epoch % 20 == 0:
                saver.save(sess, os.path.join(MODEL_DIR, "model"), global_step=epoch)
        fetchworker.shutdown()

def train_one_epoch(sess, ops, fetchworker, train_writer):
    loss_sum = []
    fetch_time = 0
    for batch_idx in range(fetchworker.num_batches):
        start = time.time()
        batch_input_data, batch_data_gt, radius =fetchworker.fetch()
        end = time.time()
        fetch_time+= end-start
        feed_dict = {ops['pointclouds_pl']: batch_input_data,
                     ops['pointclouds_gt']: batch_data_gt[:,:,0:3],
                     ops['pointclouds_gt_normal']:batch_data_gt[:,:,0:3],
                     ops['pointclouds_radius']: radius}
        summary,step, _, pred_val,gen_loss_emd = sess.run( [ops['pretrain_merged'],ops['step'],ops['pre_gen_train'],
                                                            ops['pred'], ops['gen_loss_emd']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum.append(gen_loss_emd)

        if step%30 == 0:
            pointclouds_image_input = pc_util.point_cloud_three_views(batch_input_data[0,:,0:3])
            pointclouds_image_input = np.expand_dims(np.expand_dims(pointclouds_image_input,axis=-1),axis=0)
            pointclouds_image_pred = pc_util.point_cloud_three_views(pred_val[0, :, :])
            pointclouds_image_pred = np.expand_dims(np.expand_dims(pointclouds_image_pred, axis=-1), axis=0)
            pointclouds_image_gt = pc_util.point_cloud_three_views(batch_data_gt[0, :, 0:3])
            pointclouds_image_gt = np.expand_dims(np.expand_dims(pointclouds_image_gt, axis=-1), axis=0)
            feed_dict ={ops['pointclouds_image_input']:pointclouds_image_input,
                        ops['pointclouds_image_pred']: pointclouds_image_pred,
                        ops['pointclouds_image_gt']: pointclouds_image_gt,
                        }
            summary = sess.run(ops['image_merged'],feed_dict)
            train_writer.add_summary(summary,step)

    loss_sum = np.asarray(loss_sum)
    log_string('step: %d mean gen_loss_emd: %f\n' % (step, round(loss_sum.mean(),4)))
    print 'read data time: %s mean gen_loss_emd: %f' % (round(fetch_time,4), round(loss_sum.mean(),4))


def prediction_whole_model(data_folder=None,show=False,use_normal=False):
    data_folder = '../data/test_data/our_collected_data/MC_5k'
    phase = data_folder.split('/')[-2]+data_folder.split('/')[-1]
    save_path = os.path.join(MODEL_DIR, 'result/' + phase)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    samples = glob(data_folder + "/*.xyz")
    samples.sort(reverse=True)
    input = np.loadtxt(samples[0])

    if use_normal:
        pointclouds_ipt = tf.placeholder(tf.float32, shape=(1, input.shape[0], 6))
    else:
        pointclouds_ipt = tf.placeholder(tf.float32, shape=(1, input.shape[0], 3))
    pred, _ = MODEL_GEN.get_gen_model(pointclouds_ipt, is_training=False, scope='generator', bradius=1.0,
                                      reuse=None, use_normal=use_normal, use_bn=False, use_ibn=False, bn_decay=0.95, up_ratio=UP_RATIO)
    saver = tf.train.Saver()
    _, restore_model_path = model_utils.pre_load_checkpoint(MODEL_DIR)
    print restore_model_path

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, restore_model_path)
        samples = glob(data_folder+"/*.xyz")
        samples.sort()
        total_time = 0
        for i,item in enumerate(samples):
            input = np.loadtxt(item)
            gt = input

            # input = data_provider.jitter_perturbation_point_cloud(np.expand_dims(input,axis=0),sigma=0.003,clip=0.006)
            input = np.expand_dims(input, axis=0)

            if not use_normal:
                input = input[:,:,0:3]
                gt = gt[:,0:3]
            print item, input.shape

            start_time = time.time()
            pred_pl = sess.run(pred, feed_dict={pointclouds_ipt: input})
            total_time +=time.time()-start_time
            norm_pl = np.zeros_like(pred_pl)

            ##--------------visualize predicted point cloud----------------------
            path = os.path.join(save_path,item.split('/')[-1])
            if show:
                f,axis = plt.subplots(3)
                axis[0].imshow(pc_util.point_cloud_three_views(input[:,0:3],diameter=5))
                axis[1].imshow(pc_util.point_cloud_three_views(pred_pl[0,:,:],diameter=5))
                axis[2].imshow(pc_util.point_cloud_three_views(gt[:,0:3], diameter=5))
                plt.show()
            data_provider.save_pl(path, np.hstack((pred_pl[0, ...],norm_pl[0, ...])))
            path = path[:-4]+'_input.xyz'
            data_provider.save_pl(path, input[0])
        print total_time/20

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    if PHASE=='train':
        # copy the code
        assert not os.path.exists(os.path.join(MODEL_DIR, 'code/'))
        os.makedirs(os.path.join(MODEL_DIR, 'code/'))
        os.system('cp -r * %s' % (os.path.join(MODEL_DIR, 'code/')))  # bkp of model def

        train(assign_model_path=ASSIGN_MODEL_PATH)
        LOG_FOUT.close()
    else:
        prediction_whole_model()
