# Language Translation

In this project, you’re going to take a peek into the realm of neural network machine translation. You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.

## Get the Data

Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.

## Explore the Data

Play around with view_sentence_range to view different parts of the data.

## Implement Preprocessing Function

### Text to Word Ids

As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function text_to_ids(), you'll turn source_text and target_text from words to ids. However, you need to add the <EOS> word id at the end of target_text. This will help the neural network predict when the sentence should end.

You can get the <EOS> word id by doing:

    target_vocab_to_int['<EOS>']

You can get other word ids using source_vocab_to_int and target_vocab_to_int.

### Preprocess all the data and save it

Running the code cell below will preprocess all the data and save it to file.

# Check Point

This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.

### Check the Version of TensorFlow and Access to GPU

This will check to make sure you have the correct version of TensorFlow and access to a GPU

## Build the Neural Network

You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:

- model_inputs
- process_decoder_input
- encoding_layer
- decoding_layer_train
- decoding_layer_infer
- decoding_layer
- seq2seq_model

### Input

Implement the model_inputs() function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.
- Target sequence length placeholder named "target_sequence_length" with rank 1
- Max target sequence length tensor named "max_target_len" getting its value from applying tf.reduce_max on the target_sequence_length placeholder. Rank 0.
- Source sequence length placeholder named "source_sequence_length" with rank 1

Return the placeholders in the following the tuple (input, targets, learning rate, keep probability, target sequence length, max target sequence length, source sequence length)

### Process Decoder Input

Implement process_decoder_input by removing the last word id from each batch in target_data and concat the GO ID to the begining of each batch.

### Encoding

Implement encoding_layer() to create a Encoder RNN layer:

- Embed the encoder input using tf.contrib.layers.embed_sequence 
- Construct a stacked tf.contrib.rnn.LSTMCell wrapped in a tf.contrib.rnn.DropoutWrapper 
- Pass cell and embedded input to tf.nn.dynamic_rnn() 

### Decoding - Training

Create a training decoding layer:

- Create a tf.contrib.seq2seq.TrainingHelper 
- Create a tf.contrib.seq2seq.BasicDecoder 
- Obtain the decoder outputs from tf.contrib.seq2seq.dynamic_decode 

### Decoding - Inference

Create inference decoder:

- Create a tf.contrib.seq2seq.GreedyEmbeddingHelper 
- Create a tf.contrib.seq2seq.BasicDecoder 
- Obtain the decoder outputs from tf.contrib.seq2seq.dynamic_decode 

## Build the Decoding Layer 

Implement decoding_layer() to create a Decoder RNN layer.

- Embed the target sequences
- Construct the decoder LSTM cell (just like you constructed the encoder cell above)
- Create an output layer to map the outputs of the decoder to the elements of our vocabulary
- Use the your decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob) function to get the training logits.
- Use your decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob) function to get the inference logits.

Note: You'll need to use tf.variable_scope to share variables between training and inference.



