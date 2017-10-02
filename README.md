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
