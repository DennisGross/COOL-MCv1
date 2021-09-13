from src.interpreter.data.data_generator import DataGenerator
from src.interpreter.data.data_preprocessor import *
from src.interpreter.attention_auto_encoder.training import *
from src.interpreter.attention_auto_encoder.autoencoder import *
from src.interpreter.attention_auto_encoder.translate import *
import tensorflow as tf
import os
import json

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load
from six import StringIO

from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus


class Interpreter:

    def __init__(self, project):
        self.project = project
        self.autoencoder_data_path = os.path.join(self.project.project_folder_path, 'autoencoder_data.txt')
        self.autoencoder_path = os.path.join(self.project.project_folder_path, 'autoencoder')
        self.epochs = self.project.report['num_supervised_epochs']
        self.autoencoder = None


    def decision_tree(self):
        # Preprocess Data
        generator = DataGenerator(self.project)
        generator.label_each_row_with_rl_agent_action()
        # Load data
        df = generator.df
        X = df[generator.features]
        y = df['action']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        #  Create Decision Tree classifer object
        clf = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        print(X_test, y_pred)
        # Model Accuracy
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,  
                        filled=True, rounded=True,
                        special_characters=True,feature_names = generator.features,class_names=sorted(df.action.unique().tolist()))
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(os.path.join(self.project.project_folder_path, 'decision_tree.png'))
        Image(graph.create_png())
        dump(clf, os.path.join(self.project.project_folder_path, 'decision_tree.joblib')) 



    def train_autoencoder(self):
        embedding_dim = 256
        units = 1024
        # Generate data
        generator = DataGenerator(self.project)
        generator.label_each_row_with_rl_agent_action()
        generator.generate_dataset(self.autoencoder_data_path)
        # Data Preprocessing
        m_preprocessor = DataPreprocessor(self.autoencoder_data_path)

        # Training
        train_translator = TrainTranslator(
            embedding_dim, units,
            input_text_processor=m_preprocessor.input_text_processor,
            output_text_processor=m_preprocessor.output_text_processor)

        # Configure the loss and optimizer
        train_translator.compile(
            optimizer=tf.optimizers.Adam(),
            loss=MaskedLoss(),
        )

        batch_loss = BatchLogs('batch_loss')
        train_translator.fit(m_preprocessor.dataset, epochs=self.epochs,
                            callbacks=[batch_loss])

        translator = Translator(
            encoder=train_translator.encoder,
            decoder=train_translator.decoder,
            input_text_processor=m_preprocessor.input_text_processor,
            output_text_processor=m_preprocessor.output_text_processor,
        )


        tf.saved_model.save(translator, self.autoencoder_path, signatures={'serving_default': translator.tf_translate})


    def load_autoencoder(self):
        self.autoencoder = tf.saved_model.load(self.autoencoder_path)


    def get_attention_map_for_attention_input_str(self, attention_input_str):
        m_preprocessor = DataPreprocessor(self.autoencoder_data_path)
        attention_input_str = self.command_line_state_to_autoencoder_input(attention_input_str)
        input_text = tf.constant([
            str(attention_input_str), # "It's really cold here."
        ])
        result = self.autoencoder.tf_translate(input_text = input_text)
        i = 0
        plot_attention(result['attention'][i], input_text[i], result['text'][i], m_preprocessor)


    def command_line_state_to_autoencoder_input(self, state):
        raw_features = state.split(',')
        features = []
        state_dict = {}
        tmp_str = ''
        for raw_feature in raw_features:
            key = raw_feature.split('=')[0]
            features.append(key)
            value = raw_feature.split('=')[1]
            state_dict[key] = value
        for feature in features:
            tmp_str += feature + str(int(state_dict[feature])) + ' '
        tmp_str = tmp_str.strip()
        return tmp_str






    
    
