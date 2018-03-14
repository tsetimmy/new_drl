import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
import uuid

class mlp_env_modeler:
    def __init__(self, output_size, batch_norm=True):
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.scope = str(uuid.uuid4())
        self.reuse = None

    def build(self, states, actions):
        states_embed = self.fully_connected(states, self.scope+'/states_embed')
        actions_embed = self.fully_connected(actions, self.scope+'/actions_embed')
        concat = tf.concat([states_embed, actions_embed], axis=-1)
        hidden = self.fully_connected(concat, self.scope+'/hidden')
        output = slim.fully_connected(hidden, self.output_size, activation_fn=None, scope=self.scope+'/output', reuse=self.reuse)
        self.reuse = True
        return output

    def fully_connected(self, inputs, scope):
        outputs = slim.fully_connected(inputs, 128, activation_fn=(None if self.batch_norm else tf.nn.relu), scope=scope, reuse=self.reuse)
        if self.batch_norm:
            outputs = tflearn.layers.normalization.batch_normalization(outputs, scope=scope+'_bn', reuse=self.reuse)
            outputs = tf.nn.relu(outputs)
        return outputs

    def get_losses(self, states, targets, actions):
        predictions = self.build(states, actions)
        return tf.reduce_mean(tf.reduce_sum(tf.square(predictions - targets), axis=-1)),\
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

def main():
    mlp = dmlac()

if __name__ == '__main__':
    main()
