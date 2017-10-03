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

### Build the Decoding Layer 

Implement decoding_layer() to create a Decoder RNN layer.

- Embed the target sequences
- Construct the decoder LSTM cell (just like you constructed the encoder cell above)
- Create an output layer to map the outputs of the decoder to the elements of our vocabulary
- Use the your decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob) function to get the training logits.
- Use your decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob) function to get the inference logits.

Note: You'll need to use tf.variable_scope to share variables between training and inference.

### Build the Neural Network

Apply the functions you implemented above to:

- Apply embedding to the input data for the encoder.
- Encode the input using your encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,  source_sequence_length, source_vocab_size, encoding_embedding_size).
- Process target data using your process_decoder_input(target_data, target_vocab_to_int, batch_size) function.
- Apply embedding to the target data for the decoder.
- Decode the encoded input using your decoding_layer(dec_input, enc_state, target_sequence_length, max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, dec_embedding_size) function.

## Neural Network Training

### Hyperparameters

Tune the following parameters:

- Set epochs to the number of epochs.
- Set batch_size to the batch size.
- Set rnn_size to the size of the RNNs.
- Set num_layers to the number of layers.
- Set encoding_embedding_size to the size of the embedding for the encoder.
- Set decoding_embedding_size to the size of the embedding for the decoder.
- Set learning_rate to the learning rate.
- Set keep_probability to the Dropout keep probability
- Set display_step to state how many steps between each debug output statement

### Build the Graph

Build the graph using the neural network you implemented.

Batch and pad the source and target sequences

### Train

Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.

### Save Parameters

Save the batch_size and save_path parameters for inference.

# Checkpoint

## Sentence to Sequence

To feed a sentence into the model for translation, you first need to preprocess it. Implement the function sentence_to_seq() to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using vocab_to_int
- Convert words not in the vocabulary, to the <UNK> word id.

## Translate

This will translate translate_sentence from English to French.

## Imperfect Translation

You might notice that some sentences translate better than others. Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words. For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the WMT10 French-English corpus. This dataset has more vocabulary and richer in topics discussed. However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided. Just make sure you play with the WMT10 corpus after you've submitted this project.

## Submitting This Project

When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
