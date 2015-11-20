import pandas as pd
from datetime import datetime, timedelta


class Data(object):

    def __init__(self, tracks_path):
        # Parse tracks file, ignore tracks that has no track id
        self.tracks = pd.read_csv("dataset/%s" % tracks_path, nrows=None, sep='\t', header=None,
                                  names=['user_id', 'timestamp', 'track_id', 'track_name'],
                                  parse_dates=[1], usecols=[0, 1, 4, 5], error_bad_lines=False,
                                  warn_bad_lines=True).dropna()
        self.experiments = {}

    def create_experiment_data(self, training_length, testing_length, finish_testing, experiment_id=None):
        if experiment_id is None:
            experiment_id = len(self.experiments)
        begin_testing = finish_testing - timedelta(days=testing_length)
        begin_training = begin_testing - timedelta(days=training_length)
        tracks_training = self.tracks[(self.tracks['timestamp'] >= begin_training) &
                                      (self.tracks['timestamp'] < begin_testing)]
        tracks_testing = self.tracks[(self.tracks['timestamp'] >= begin_testing) &
                                     (self.tracks['timestamp'] < finish_testing)]

        self.experiments[experiment_id] = {'training': tracks_training, 'testing': tracks_testing}
        return experiment_id

    def get_training_data(self, experiment_id):
        return self.experiments[experiment_id]['training']

    def get_testing_data(self, experiment_id):
        return self.experiments[experiment_id]['testing']
