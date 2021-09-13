import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_text as tf_text
import pathlib

class DataPreprocessor:
    
    def __init__(self, path):
        targ, inp = self.__load_data(path)
        self.BUFFER_SIZE = len(inp)
        self.BATCH_SIZE = 64
        self.dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(self.BUFFER_SIZE)
        self.dataset = self.dataset.batch(self.BATCH_SIZE)
        max_vocab_size = 5000
        self.input_text_processor = preprocessing.TextVectorization(
            standardize=self.tf_lower_and_split_punct,
            max_tokens=max_vocab_size)
        self.input_text_processor.adapt(inp)

        self.output_text_processor = preprocessing.TextVectorization(
            standardize=self.tf_lower_and_split_punct,
            max_tokens=max_vocab_size)

        self.output_text_processor.adapt(targ)

    def __load_data(self, path_to_file):
        path = pathlib.Path(path_to_file)
        text = path.read_text(encoding='utf-8')

        lines = text.splitlines()
        pairs = [line.split('\t') for line in lines]

        inp = [inp for inp, targ in pairs]
        targ = [targ for inp, targ in pairs]

        return targ, inp

    def tf_lower_and_split_punct(self, text):
        # Split accecented characters.
        #text = tf_text.normalize_utf8(text, 'NFKD')
        text = tf.strings.lower(text)
        # Keep space, a to z, and select punctuation.
        #text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
        # Add spaces around punctuation.
        text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
        # Strip whitespace.
        text = tf.strings.strip(text)

        text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
        return text
