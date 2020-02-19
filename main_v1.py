
import tensorflow as tf
# from  model import SRCNN
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_steps", 100000, "Number of epoch")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("learning_rate", 0.0002, "learning rate")
flags.DEFINE_integer("image_dim", 0.0002, "image dim")
flags.DEFINE_integer("image_dim", 0.0002, "image dim")


