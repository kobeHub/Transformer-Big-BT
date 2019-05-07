"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Wed 01 May 2019 09:11:20 PM CST

 Define the Transformer model with encoder, decoder stack
"""

import tensorflow as tf
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from model import attention
from model import beam_search
from model import embedding_layers
from model import ffn
from model import model_utils

from preprocess.tokenizer import EOS_ID

_NEG_NIF = -1e9



class Transformer(object):
    """Transformer model for seq2seq data.
    
    The basic model consists of an encoder and decoder. The input is int sequence
    . The encoder produces a continous representation , the decoder use encoder output
    to generate probabilities for the output sequence.
    """

    def __init__(self, params, trainable):
        self.trainable = trainable
        self.params = params

        self.embedding_softmax_layer = embedding_layers.EmbeddingSharedWeights(
                params['vocab_size'], 
                params['hidden_size'])
        self.encoder_stack = EncoderStack(params, trainable)
        self.decoder_stack = DecoderStack(params, trainable)


    def __call__(self, inputs, targets=None):
        """Calculate target logits or inferred target seq.
        
        Args:
            inputs: input tensor [batch_size, input_length]
            targets: int tensor [batch_size, target_length]

        @r:
            If targets is defined, then return logits for each word in the target
            sequence. float tensor with shape [batch_size, target_length, vocab_size]
            If target is none, then generate output sequence one token at a time.
            returns a dictionary {
            output: [batch_size, decoded length]
            score: [batch_size, float]}
        """
        initializer = tf.variance_scaling_initializer(
                self.params['initializer_gain'], mode='fan_avg', distribution='uniform')
        with tf.variable_scope('Transformer', initializer=initializer):
            # Attention bias for encoder self-attention and decoder multi-head
            # attention layers
            attention_bias = model_utils.get_padding_bias(inputs)

            # Run inputs through encoder layers to map symbol representations
            # to continuous representations
            encoder_outputs = self.encode(inputs, attention_bias)

            if targets is None:
                return self.predict(encoder_output, attention_bias)
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias)
                return logits


    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs.
        
        Args:
            inputs: int tensor [batch_size, input_length]
            attention_bias: float tensor [batch_size, 1, 1, input_length]

        @r:
            float tensor [batch_size, input_length, hidden_size]
        """
        with tf.name_scope('encode'):
            # Prepare inputs to the layer stack by adding position encodings
            # and applying dropout
            embeded_inputs = self.embedding_softmax_layer(inputs)
            inputs_padding = model_utils.get_padding(inputs)

            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(inputs)[1]
                pos_encoding = model_utils.position_encoding(length,
                        self.params['hidden_size'])
                encoder_inputs = embeded_inputs + pos_encoding

            if self.trainable:
                encoder_inputs = tf.nn.dropout(
                        encoder_inputs, rate=self.params['layers_postprocess_dropout'])

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, targets, encoder_outputs, attention_bias):
        """Generate logits for each value in the target sequence.
        
        Args:
            target: int tensor [batch_size, target_length]
            encoder_outputs: continuous representation of input seq.
                float tensor  [batch_size, input_length, hidden_size]
            attention_bias: float tensor [batch_size, 1, 1, input_length]

        @r:
            float32 [batch_size, target_length, vocab_size]
        """
        with tf.name_scope('decode'):
            decoder_inputs = self.embedding_softmax_layer(targets)
            with tf.name_scope('shift_targets'):
                # Shitf targets to right, remove the last element
                decoder_inputs = tf.pad(decoder_inputs,
                        [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += model_utils.position_encoding(length,
                        self.params['hidden_size'])
            if self.trainable:
                decoder_inputs =  tf.nn.dropout(decoder_inputs,
                        rate=self.params['layers_postprocess_dropout'] )

            # Run values
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length)
            outputs = self.decoder_stack(decoder_inputs,
                    encoder_output,
                    decoder_self_attention_bias,
                    attention_bias)
            logits = self.embedding_softmax_layer.linear(output)

            return logits


    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Return a decoding function that calculates logits of the next tokens."""
        timing_signal = model_utils.position_encoding(max_decode_length + 1,
               self.params['hidden_size'] )
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
            
            Args:
                ids: current decoded seq. int [batch_size * beam_size, i + 1]
                i: loop index
                cache: dict of values storing in the encoder output,
                        encode-decoder attention bias and previous decoder
                        attention values

            @r:
                Tuple of (logits [batch_size * beam_size, vocab_size],
                new cache values)
            """
            decoder_input = ids[:, -1:]
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i+1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i+1, :i+1]
            decoder_outputs = self.decoder_stack(
                    decoder_input,
                    cache.get('encoder_outputs'),
                    self_attention_bias,
                    cache.get('encoder_decoder_attention_bias'),
                    cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits.squeeze(logits, axis=[1])
            return logits, cache
        return symbols_to_logits_fn


    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params['extra_decode_length']

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        # Create initial ids 
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create initial set of IDs that will ne passed into symbols_to_logits_fn
        cache = {
                'layer_{}'.format(layer): {
                    'k': tf.zeros([batch_size, 0, self.params['hidden_size']]),
                    'v': tf.zeros([batch_size, 0, self.params['hidden_size']]),
                    } for layer in range(self.params['num_hidden_layers'])
                }

        cache['encoder_outputs'] = encoder_outputs
        cache['encoder_decoder_attention_bias'] = encoder_decoder_attention_bias

        # Beam search to find top beam_size seq and scores
        decoded_ids, scores = beam_search.sequence_beam_search(
                symbols_to_logits_fn,
                initial_ids,
                cache,
                self.params['vocab_size'],
                self.params['beam_size'],
                self.params['alpha'],
                max_decode_length,
                EOS_ID)

        # Top seq 
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {'outputs': top_decoded_ids, 'scores': top_scores}



class BatchNormalization(tf.layers.Layer):
    """BN layer"""
    
    def __init__(self, hidden_size):
        super(BatchNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale =  tf.get_variable('batch_norm_scale',
                [self.hidden_size],
                initializer=tf.ones_initializer())
        self.bias = tf.get_variable('batch_norm_bias',
                [self.hidden_size],
                initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias




class PrePostProcessingWrapper(object):
    """Wrapper class applies pre and post processing."""

    def __init__(self, layer, params, trainable):
        self.layer = layer
        self.dropout = params['layers_postprocess_dropout']
        self.trainable = trainable

        self.layer_norm = BatchNormalization(params['hidden_size'])

    def __call__(self, x, *args, **kwargs):
        y = self.layer_norm(x)
        y = self.layer(y, *args, **kwargs)

        if self.trainable:
            y = tf.nn.dropout(y, rate=self.dropout)
        return x + y    # residual link




class EncoderStack(tf.layers.Layer):
    """Transformer encoder stack.
    Which is made of N layers, each layer is composed od the sublayers:
     1. self-attention layer
     2. ffn (2-layers)
    """

    def __init_(self, params, trainable):
        super(EncoderStack, self).__init__()
        self.layers = []

        for _ in range(params['num_hidden_layers']):
            self_attention_layer = attention.SelfAttention(
                    params['hidden_size'],
                    params['num_heads'],
                    params['attention_dropout'],
                    trainable)
            feed_forward_net = ffn.FeedForwardNetwork(
                    params['hidden_size'],
                    params['filter_size'],
                    params['relu_dropout'],
                    trainable,
                    param['allow_ffn_pad'])

            self.layers.append([
                    PrePostProcessingWrapper(self_attention_layer, params, trainable),
                    PrePostProcessingWrapper(feed_forward_net, params, trainable)])

        # Final norm layer
        self.output_normalization = BatchNormalization(params['hidden_size'])


    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """Return outputs of the encoder layer.
        
        Args:
            encoder_inputs: float32 [batch_size, input_length, hidden_size]
            attention_bias: bias for the self_attention_layer [batch_size, 1, 1, input_length]

        @r:
            float32 [batch_size, input_length, hidden_size]
        """
        for i, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            ffn_ = layer[1]

            with tf.variable_scope('layer_{}'.format(i)):
                with tf.variable_scope('self_attention'):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
                with tf.variable_scope('ffn'):
                    encoder_inputs = ffn_(encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)




class DecoderStack(tf.layers.Layer):
    """Transformer decode stack.
    
    Decoder stack is made of N layers. Each layer is composed of 
    the sublayers:
        1. self_attention layer
        2. Multi-headed attention combining encoder outputs and 
            previous self_attention_layer results
        3. ffn (2-layers)
    """

    def __init__(self, params, trainable):
        super(DecoderStack, self).__init__()
        self.layers = []
        for _ in range(params['num_hidden_layers']):
            self_attention_layer = attention.SelfAttention(
                    params['hidden_size'],
                    params['num_heads'],
                    params['attention_dropout'],
                    trainable)
            enc_dec_atten_layer = attention.Attention(
                    params['hidden_size'],
                    params['num_heads'],
                    params['attention_dropout'],
                    trainable)
            ffn_ = ffn.FeedForwardNetwork(
                    params['hidden_size'],
                    params['filter_size'],
                    params['relu_dropout'],
                    trainable,
                    params['allow_ffn_pad'])
            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, trainable),
                PrePostProcessingWrapper(enc_dec_atten_layer, params, trainable),
                PrePostProcessingWrapper(ffn_, params, trainable)])

        # Outputs of the decoder satckj
        self.output_normalization = BatchNormalization(params['hidden_size'])

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
            attention_bias, cache=None):
        """Return the output of encode stack.
        
        Args:
            decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
            encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
            decoder_self_attention_bias: bias for decoder self-attention layer.
                                        [1, 1, target_len, target_length]
            attention_bias: bias for encoder-decoder attention layer.
                                        [batch_size, 1, 1, input_length]
            cache: (Used for fast decoding) A nested dictionary storing previous
                     decoder self-attention values. The items are:
                    {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]},
                        ...}
        @r:
            Output of decoder layer stack.
            float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for i, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_atten_layer = layer[1]
            ffn_ = layer[2]

            # Run inputs through layers
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache else None

            with tf.variable_scope(layer_name):
                with tf.variable_scope('self_attention_layer'):
                    decoder_inputs = self_attention_layer(
                            decoder_inputs,
                            decoder_self_attention_bias,
                            cache=layer_cache)
                with tf.variable_scope('enc_dec_atten_layer'):
                    decoder_inputs = enc_dec_atten_layer(
                            decoder_inputs,
                            encoder_outputs,
                            attention_bias)
                with tf.variable_scope('ffn'):
                    decoder_inputs = ffn_(decoder_inputs)

        return self.output_normalization(decoder_inputs)
        
