import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
auctionmatch_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_auctionmatch_so.so'))

def auction_match(xyz1,xyz2):
	'''
input:
	xyz1 : batch_size * #points * 3
	xyz2 : batch_size * #points * 3
returns:
	matchl : batch_size * #npoints
	matchr : batch_size * #npoints
	'''
	return auctionmatch_module.auction_match(xyz1,xyz2)
ops.NoGradient('AuctionMatch')

# TF1.0 API requires set shape in C++
# @tf.RegisterShape('AuctionMatch')
# def _auction_match_shape(op):
# 	shape1=op.inputs[0].get_shape().with_rank(3)
# 	shape2=op.inputs[1].get_shape().with_rank(3)
# 	return [
# 		tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
# 		tf.TensorShape([shape2.dims[0],shape2.dims[1]])
# 	]

if __name__=='__main__':
    from tf_ops.grouping import tf_grouping
    from tf_ops.sampling import tf_sampling

    npoint=4096
    xyz1_in=tf.placeholder(tf.float32,shape=(32,npoint,3))
    xyz2_in=tf.placeholder(tf.float32,shape=(32,npoint,3))
    matchl_out,matchr_out=auction_match(xyz1_in,xyz2_in)
    matched_out=tf_sampling.gather_point(xyz2_in,matchl_out)
    import numpy as np
    np.random.seed(100)
    xyz1=np.random.randn(32,npoint,3).astype('float32')
    xyz2=xyz1.copy()+np.random.randn(32,npoint,3)*0.01
    for i in xrange(len(xyz2)):
        xyz2[i]=np.roll(xyz2[i],i,axis=0)
    with tf.Session('') as sess:
        ret=sess.run(matched_out,feed_dict={xyz1_in:xyz1,xyz2_in:xyz2})
    print ((xyz1-ret)**2).mean()
