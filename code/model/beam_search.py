"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Tue 30 Apr 2019

 Implement the beam search to explore the translated sequence with
 highest probability. 
"""

import tensorflow as tf
from tensorflow.python.util import nest

INF = 1. * 1e7



class _StateKeys:
    """Key to beam search loop state"""

    CUR_INDEX = 'CUR_INDEX'

    # Top sequences are alive for each batch item. Alive seq
    # have not generated an EOS token. Seq reach EOS are marked
    # as finished and moved to FINISHED_SEQ tensor
    ALIVE_SEQ = 'ALIVE_SEQ'                     # shape [batch_size, beam_size, CUR_INDEX + 1]
    ALIVE_SEQ_PROBS = 'ALIVE_SEQ_PROBS'
    ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
    # Dict for cached values for each alive seq. The cache stroes 
    # the encoder output, attention bias, decoder attention output
    # from the previous iteration
    ALIVE_CACHE = 'ALIVE_CACHE'

    # Top finished seq for each batch item.
    # shape [batch_size, beam_size, CUR_INDEX + 1], seqs shorter than 
    # CUR_INDEX + 1 are padding with 0
    FINISHED_SEQ = 'FINISHED_SEQ'

    # Scores for each finished sequence. Score = log probability / length norm
    # Shape [batch_size, beam_size]
    FINISHED_SCORES = "FINISHED_SCORES"
    # Flags indicating which sequences in the finished sequences are finished.
    # At the beginning, all of the sequences in FINISHED_SEQ are filler values.
    # True -> finished sequence, False -> filler. Shape [batch_size, beam_size]
    FINISHED_FLAGS = "FINISHED_FLAGS"



class SequenceBeamSearch:
    def __init__(self, symbols_to_logits_fn, vocab_size, batch_size,
            beam_size, alpha, max_decode_length, eos_id):
        self.symbols_to_logits_fn = symbols_to_logits_fn
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_decode_length = max_decode_length
        self.eos_id = eos_id

    def _create_init_state(self, initial_ids, initial_cache):
        """Get the initial state dictonary and its shape invariants.
        
        Args:
            initial_ids: initial ids passed to symbols_to_logits_fn . [batch_size, 1]
            initial_cache: dict stores value to be passed into symbols_to_logits_fn.

        @r:
            state and shape invariant dict with key from _StateKey
        """
        cur_index = tf.constant(0)

        # Alive seq shape [batch_size, beam_size, 1] 
        alive_seq = _expand_to_beam_size(initial_ids, self.beam_size)
        alive_seq = tf.expand_dims(alive_seq, axis=2)

        # tensor to store initial log probability
        initial_log_probs = tf.constant([[0.] + [-float('inf')] * 
            (self.beam_size - 1)])
        alive_log_probs = tf.tile(initial_log_probs, [self.batch_size, 1])

        # Expand all the value stored in the dict to beam_size, so that each 
        # beam has a separate cache   
        alive_cache = nest.map_structure(
                lambda t: _expand_to_beam_size(t, self.beam_size), initial_cache)


        # initialize tensor storing finished seq
        finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)
        finished_scores = tf.ones([self.batch_size, self.beam_size]) * -INF

        # init the finished flags with False value
        finished_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)

        state = {
                _StateKeys.CUR_INDEX: cur_index,
                _StateKeys.ALIVE_SEQ: alive_seq,
                _StateKeys.ALIVE_SEQ_PROBS: alive_log_probs,
                _StateKeys.ALIVE_CACHE: alive_cache,
                _StateKeys.FINISHED_SEQ: finished_seq,
                _StateKeys.FINISHED_SCORES: finished_scores,
                _StateKeys.FINISHED_FLAGS: finished_flags,
                }
        # Create state invariants for each value in the state dictionary. Each
        # dimension must be a constant or None. A None dimension means either:
        #   1) the dimension's value is a tensor that remains the same but may
        #      depend on the input sequence to the model (e.g. batch size).
        #   2) the dimension may have different values on different iterations.
        state_shape_invariants = {
            _StateKeys.CUR_INDEX: tf.TensorShape([]),
            _StateKeys.ALIVE_SEQ: tf.TensorShape([None, self.beam_size, None]),
            _StateKeys.ALIVE_LOG_PROBS: tf.TensorShape([None, self.beam_size]),
            _StateKeys.ALIVE_CACHE: nest.map_structure(
                _get_shape_keep_last_dim, alive_cache),
            _StateKeys.FINISHED_SEQ: tf.TensorShape([None, self.beam_size, None]),
            _StateKeys.FINISHED_SCORES: tf.TensorShape([None, self.beam_size]),
            _StateKeys.FINISHED_FLAGS: tf.TensorShape([None, self.beam_size])
            }

        return state, state_shape_invariants


    def _continue_search(self, state):
        """Define whether continue to search.
        
        The search loop will stop when:
        1. decode length has been reached; or
        2. when the worst score in the finished_seq is better than 
                the best score in the alive_seq (finished_seq are unchanging)

        Args:
            state: dict with current loop state

        @r:
            bool tensor
        """
        index = state[_StateKeys.CUR_INDEX]
        alive_log_probs = state[_StateKeys.ALIVE_SEQ_PROBS]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        not_max_decode_length = tf.less(index, self.max_decode_length)

        # Calcute largest length penalty 
        max_length_norm = _length_normalization(self.alpha, self.max_decode_length)
        best_alive_scores = alive_log_probs[:, 0] / max_length_norm

        # Compute the worst score in finished seq for each batch element
        finished_scores *= tf.cast(finished_flags, tf.float32)
        worst_score = tf.reduce_min(finished_seq, axis=1)

        # If there are no finished seqs then set worst_score = -INF
        finished_batches = tf.reduce_any(finished_flags, 1)     # logical `or` operation along axis 1
        worst_score += (1. - tf.cast(finished_batches, tf.float32)) * -INF

        worst_finished_score_better_than_best_alive_score = tf.reduce_all(
                tf.greater(worst_socre, best_alive_scores)
                )

        return tf.logical_and(
                not_max_decode_length,
                worst_finished_score_better_than_best_alive_score
                )


    def grow_alive_seq(self, state):
        """Grow alive sequence by one token, and collect the top 2*beami_size seq
        
        Args:
            state: dict of current loop state
        @r:
        Tuple of (top 2*beam_size seq [batch_size, 2*beam_size, cur_index + 1],
            scores of the seq [batch_size, 2 * beam_size])
            new alive cache
        """
        index = state[_StateKeys.CUR_INDEX]
        alive_seq = state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = state[_StateKeys.ALIVE_SEQ_PROBS]
        alive_cache = state[_StateKeys.ALIVE_CACHE]

        beams_keep = 2 * beam_size

        flat_ids = flatten_beam_dim(alive_seq)    
        flat_cache = nest.map_structure(_flatten_beam_dim, alive_cache)

        flat_logits, flat_cache = self.symbols_to_logits_fn(flat_ids, index, flat_cache)

        # Unflatten logits to shape [batch_size, beam_size, vocab_size]
        logits = _unflatten_beam_dim(flat_logits, self.batch_size, self.beam_size)
        new_cache = nest.map_structure(
                lambda t: _unflatten_beam_dim(t, self.batch_size, self.beam_size),
                flat_cache)

        # Convert logits to normalized log probs   
        candidate_log_probs = _log_prob_from_logits(logits)

        # Calculate new log probabilities if each of the alive sequences were
        # extended # by the the candidate IDs.
        # Shape [batch_size, beam_size, vocab_size]
        log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

        # Each batch item has beam_size * vocab_size candidate sequences. For each
        # batch item, get the k candidates with the highest log probabilities.
        flat_log_probs = tf.reshape(log_probs,
                                [-1, self.beam_size * self.vocab_size])
        topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=beams_to_keep)

        # Extract the alive sequences that generate the highest log probabilities
        # after being extended.
        topk_beam_indices = topk_indices // self.vocab_size
        topk_seq, new_cache = _gather_beams(
            [alive_seq, new_cache], topk_beam_indices, self.batch_size,
            beams_keep)

        # Append the most probable IDs to the topk sequences
        topk_ids = topk_indices % self.vocab_size
        topk_ids = tf.expand_dims(topk_ids, axis=2)
        topk_seq = tf.concat([topk_seq, topk_ids], axis=2)
        
        return topk_seq, topk_log_probs, new_cache
    

    def get_new_alive_state(self, new_seq, new_log_probs, new_cache):
        """Gather the top k seq that are still alive.
        
        Args:
            new_seq: new sequence generated by growing the current alive seq
            int32 tensor shape [batch_size, 2 * beam_size,.  cur_index + 1]
            new_log_probs: Log probability of new sequence. float32
                        shape [batch_size, beam_size]
            new_cache: dict of cache value for each seq

        @r: 
            Dictionary with alive keys from _StateKeys:
                {Top beam_size sequences that are still alive (don't end with eos_id)
                Log probabilities of top alive sequences
                Dict cache storing decoder states for top alive sequences}
        """
        # To prevent finished seqs from being considered, set log probs to -INF
        new_finished_flag = tf.equal(new_seq[:, :, -1], self.eos_id)
        new_log_probs += tf.cast(new_finished_flag, tf.float32) * -INF

        # 
        top_alive_seq, top_alive_log_probs, top_alive_cache = _gather_topk_beam(
                [new_seq, new_log_probs, new_cache], 
                new_log_probs,
                self.batch_size,
                self.beam_size
                )

        return {
                _StateKeys.ALIVE_SEQ: top_alive_seq,
                _StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
                _StateKeys.ALIVE_CACHE: top_alive_cache
                }


    def get_new_finished_state(self, state, new_seq, new_log_probs):
        """Combine old and new finished seq, and gather the topk alive sequence.
        
        Args:
            state: dict of current loop state
            new_seq: new seq generated by grow_alive_seq. int32 shape [batch_size, beam_size,
                    i+1]
            new_log_probs: float32 shape [batch_size, beam_size]

        @r:
            dict with key from _StateKeys {Top beam_size finished seq based on score,
                score of finishedb seq,
                finished_flags}
        """
        i = state[_StateKeys.CUR_INDEX]
        finished_seq = state[_StateKeys.FINISHED_SEQ]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        # First append a column of 0-ids to finished_seq to increment the length.
        # New shape of finished_seq: [batch_size, beam_size, i + 1]
        finished_seq = tf.concat(
            [finished_seq,
            tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)], axis=2)

        # Calculate new seq scores from log probabilities.
        length_norm = _length_normalization(self.alpha, i + 1)
        new_scores = new_log_probs / length_norm

        # Set the scores of the still-alive seq in new_seq to large negative values.
        new_finished_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
        new_scores += (1. - tf.to_float(new_finished_flags)) * -INF

        # Combine sequences, scores, and flags.
        finished_seq = tf.concat([finished_seq, new_seq], axis=1)
        finished_scores = tf.concat([finished_scores, new_scores], axis=1)
        finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

        # Return the finished sequences with the best scores.
        top_finished_seq, top_finished_scores, top_finished_flags = (
            _gather_topk_beams([finished_seq, finished_scores, finished_flags],
                               finished_scores, self.batch_size, self.beam_size))

        return {
            _StateKeys.FINISHED_SEQ: top_finished_seq,
            _StateKeys.FINISHED_SCORES: top_finished_scores,
            _StateKeys.FINISHED_FLAGS: top_finished_flags
        }


    def _search_step(self, state):
        """Beam search loop body.
        
        Grow alive sequences by a single ID. Sequences that have reached the EOS
        token are marked as finished. The alive and finished sequences with the
        highest log probabilities and scores are returned.
    
        A sequence's finished score is calculating by dividing the log probability
        by the length normalization factor. Without length normalization, the
        search is more likely to return shorter sequences.
        
        Args:
            state: A dictionary with the current loop state.
        @r:
            new state dictionary.
        """
        new_seq, new_log_probs, new_cache = self._grow_alive_seq(state)
        # Collect top beam_size alive sequences
        alive_state = self._get_new_alive_state(new_seq, new_log_probs, new_cache)

        # Combine newly finished sequences with existing finished sequences, and
        # collect the top k scoring sequences.
        finished_state = self._get_new_finished_state(state, new_seq, new_log_probs)

        # Increment loop index and create new state dictionary
        new_state = {_StateKeys.CUR_INDEX: state[_StateKeys.CUR_INDEX] + 1}
        new_state.update(alive_state)
        new_state.update(finished_state)
        
        return [new_state]


    def search(self, initial_ids, initial_cache):
        """Beam search for sequence with highest scores"""
        state, state_shapes = self._create_init_state(initial_ids, initial_cache)

        # 
        finished_state = tf.while_loop(self._continue_search, 
                self._search_step, loop_vars=[state],
                shape_invariants=[state_shapes], parallel_iterations=1,
                back_prop=False)
        finished_state = finished_state[0]
    
        alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
        finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
        finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
        finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]
        
        finished_seq = tf.where(
                tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
        finished_scores = tf.where(
                tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
        return finished_seq, finished_scores




################################# Class utils ########################################

def sequence_beam_search(symbols_to_logits_fn, initial_ids, initial_cache,
        vocab_size, beam_size, alpha, max_decode_length, eos_id):
    
    batch_size = tf.shape(initial_ids)[0]
    sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size, 
            beam_size, alpha, max_decode_length, eos_id)
    return sbs.search(initial_ids, initial_cache)



def _log_prob_from_logits(logits):
    return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)



def _length_normalization(alpha, length):
    return tf.pow(((5. + tf.cast(length, tf.float32)) / 6.), alpha)



def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by given beam_size.
    
    The ops is same as tf.tile()
    """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)



def _shape_list(tensor):
    """Return a list of tensor's shape and ensure there is no None"""
    shape = tensor.get_shape().as_list()

    dynamic = tf.shape(tensor)
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = dynamic[i]
    return shape



def _get_shape_keep_last_dim(tensor):
    shape_list = _shape_list(tensor)

    # Only the last
    for i in range(len(shape_list) - 1):
        shape_list[i] = None

    if isinstance(shape_list[-1], tf.Tensor):
        shape_list[-1] = None
    return tf.TensorShape(shape_list)



def _flatten_beam_dim(tensor):
  """Reshapes first two dimensions in to single dimension.
  Args:
    tensor: Tensor to reshape of shape [A, B, ...]
  Returns:
    Reshaped tensor of shape [A*B, ...]
  """
  shape = _shape_list(tensor)
  shape[0] *= shape[1]
  shape.pop(1)  # Remove beam dim
  return tf.reshape(tensor, shape)



def _unflatten_beam_dim(tensor, batch_size, beam_size):
  """Reshapes first dimension back to [batch_size, beam_size].
  Args:
    tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
    batch_size: Tensor, original batch size.
    beam_size: int, original beam size.
  Returns:
    Reshaped tensor of shape [batch_size, beam_size, ...]
  """
  shape = _shape_list(tensor)
  new_shape = [batch_size, beam_size] + shape[1:]
  return tf.reshape(tensor, new_shape)



def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
  """Gather beams from nested structure of tensors.
  Each tensor in nested represents a batch of beams, where beam refers to a
  single search state (beam search involves searching through multiple states
  in parallel).
  This function is used to gather the top beams, specified by
  beam_indices, from the nested tensors.
  Args:
    nested: Nested structure (tensor, list, tuple or dict) containing tensors
      with shape [batch_size, beam_size, ...].
    beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each
     value in beam_indices must be between [0, beam_size), and are not
     necessarily unique.
    batch_size: int size of batch
    new_beam_size: int number of beams to be pulled from the nested tensors.
  Returns:
    Nested structure containing tensors with shape
      [batch_size, new_beam_size, ...]
  """
  # Computes the i'th coodinate that contains the batch index for gather_nd.
  # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..].
  batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
  batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])

  # Create coordinates to be passed to tf.gather_nd. Stacking creates a tensor
  # with shape [batch_size, beam_size, 2], where the last dimension contains
  # the (i, j) gathering coordinates.
  coordinates = tf.stack([batch_pos, beam_indices], axis=2)

  return nest.map_structure(
      lambda state: tf.gather_nd(state, coordinates), nested)



def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
  """Gather top beams from nested structure."""
  _, topk_indexes = tf.nn.top_k(score_or_log_prob, k=beam_size)
  return _gather_beams(nested, topk_indexes, batch_size, beam_size)
