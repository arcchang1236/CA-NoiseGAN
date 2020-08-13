import tensorflow as tf

def restore_last_model(ckpt_dir, sess, saver):
  last_epoch = 0
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if ckpt:
    print('loading ' + ckpt.model_checkpoint_path)
    try:
      last_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      saver.restore(sess, ckpt.model_checkpoint_path)
    except:
      print('failed to load last model, starting from epoch 1')
  return last_epoch