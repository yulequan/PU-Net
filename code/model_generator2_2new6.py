import tensorflow as tf
from utils import tf_util2
from utils.pointnet_util import pointnet_sa_module,pointnet_fp_module

def placeholder_inputs(batch_size, num_point,up_ratio = 4):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, num_point*up_ratio, 3))
    pointclouds_normal = tf.placeholder(tf.float32, shape=(batch_size, num_point * up_ratio, 3))
    pointclouds_radius = tf.placeholder(tf.float32, shape=(batch_size))
    return pointclouds_pl, pointclouds_gt,pointclouds_normal, pointclouds_radius


def get_gen_model(point_cloud, is_training, scope, bradius = 1.0, reuse=None, use_rv=False, use_bn = False,use_ibn = False,
                  use_normal=False,bn_decay=None, up_ratio = 4,idx=None):

    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        l0_xyz = point_cloud[:,:,0:3]
        if use_normal:
            l0_points = point_cloud[:,:,3:]
        else:
            l0_points = None
        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.05,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=bradius*0.1,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer3')

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.3,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers
        up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                       scope='fa_layer1',bn=use_bn,ibn = use_ibn)

        up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                       scope='fa_layer2',bn=use_bn,ibn = use_ibn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn,ibn = use_ibn)

        ###concat feature
        with tf.variable_scope('up_layer',reuse=reuse):
            new_points_list = []
            for i in range(up_ratio):
                concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=-1)
                concat_feat = tf.expand_dims(concat_feat, axis=2)
                concat_feat = tf_util2.conv2d(concat_feat, 256, [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=False, is_training=is_training,
                                              scope='fc_layer0_%d'%(i), bn_decay=bn_decay)

                new_points = tf_util2.conv2d(concat_feat, 128, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=use_bn, is_training=is_training,
                                             scope='conv_%d' % (i),
                                             bn_decay=bn_decay)
                new_points_list.append(new_points)
            net = tf.concat(new_points_list,axis=1)

        #get the xyz
        coord = tf_util2.conv2d(net, 64, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='fc_layer1', bn_decay=bn_decay)

        coord = tf_util2.conv2d(coord, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='fc_layer2', bn_decay=bn_decay,
                             activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3
        coord = tf.squeeze(coord, [2])  # B*(2N)*3

    return coord,None