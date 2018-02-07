#! /usr/bin/env python2
#-*- coding: utf-8 -*-
#
# @author wbbtiger@gmail.com
# @brief Create a softmax regression model for mnist, and export as binary protobuf.
#

import tensorflow as tf
import getopt
import sys
import exceptions
import traceback

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def usage(progname):
    print('''Usage:
    {progname} [-h | --help] --export_dir <dir> [--as_text]'''.format(progname=progname))
    
def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", [ 'help', 'export_dir=', 'as_text' ])
        except getopt.error as msg:
            raise Usage(msg)

        global g_debug_mode
        export_dir = ''
        as_text = False
        for o, a in opts:
            if o in ('-h', '--help'):
                usage(argv[0])
                return 0
            elif o in ('-d'):
                g_debug_mode = True
            elif o in ('--export_dir'):
                export_dir = a
            elif o in ('--as_text'):
                as_text = True
            else:
                raise Usage('Bad option: %s' % (o))
        if len(export_dir) == 0:
            raise Usage('Bad value for option --export_dir')

        # Create the model
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b

        # Define loss and optimizer
        y_ = tf.placeholder(tf.int32, [None], name='y_')
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name='train_step')

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.cast(y_, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # Create the global variables initializer.
        init = tf.variables_initializer(tf.global_variables(), name='init')
        
        # Export the computation graph.
        tf.train.write_graph(tf.get_default_graph().as_graph_def(), export_dir, 'mnist_graph.pb', as_text=as_text)
    except Usage as err:
        print(err.msg)
        usage(argv[0])
        return 1
    except Exception as e:
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    main()
