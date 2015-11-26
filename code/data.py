import pandas as pd
from datetime import timedelta


class Data(object):
    """
        Class that loads whole last.fm dataset on the initialization and handles
        experiment data extraction out of it.
    """
    def __init__(self, tracks_path):
        """
        Parse tracks file, ignore tracks that has no track ID.
        Keep the dictionary of all experiments, so that the data can be easily retrieved later.
        :param tracks_path: path to the dataset
        :return:
        """
        self.tracks = pd.read_csv(tracks_path, nrows=None, sep='\t', header=None,
                                  names=['user_id', 'timestamp', 'track_id', 'track_name'],
                                  parse_dates=[1], usecols=[0, 1, 4, 5], error_bad_lines=False,
                                  warn_bad_lines=True).dropna()
        self.experiments = {}

    def create_experiment_data(self, training_length, testing_length, finish_testing, experiment_id=None, partition=False):
        """
        Extract data for the experiment and save it in the dictionary.If partition is needed,
        partition training dataset into multiple datasets that keep at most testing_length days of user behaviour
        :param training_length: how many days of training data to extract
        :param testing_length: how many days of testing data to extract
        :param finish_testing: datetime object that specifies the date when the testing finishes
        :param experiment_id: int that specifies experiment id (if missing, will be given new experiment id)
        :param partition: bool varibale to decide whether the data needs to be partitioned (if missing, don't partition)
        :return: experiment_id
        """
        if experiment_id is None:
            experiment_id = len(self.experiments)
        begin_testing = finish_testing - timedelta(days=testing_length)
        begin_training = begin_testing - timedelta(days=training_length)
        tracks_training = self.tracks[(self.tracks['timestamp'] >= begin_training) &
                                      (self.tracks['timestamp'] < begin_testing)]
        tracks_testing = self.tracks[(self.tracks['timestamp'] >= begin_testing) &
                                     (self.tracks['timestamp'] < finish_testing)]
        self.experiments[experiment_id] = {'training': tracks_training,
                                           'testing': tracks_testing,
                                           'norm_scalier': testing_length*1.0/training_length}
        if partition:
            partitions = []
            while not training_length == 0:
                if training_length - testing_length >= 0:
                    partitions.append(testing_length)
                    training_length -= testing_length
                else:
                    partitions.append(training_length)
                    training_length = 0
            self.experiments[experiment_id]['training_partitions'] = []
            finish_partition = begin_testing
            for i in range(len(partitions)):
                begin_partition = finish_partition - timedelta(days=partitions[i])
                partition_data = tracks_training[(tracks_training['timestamp'] >= begin_partition) &
                                                 (tracks_training['timestamp'] < finish_partition)]
                self.experiments[experiment_id]['training_partitions'].append(partition_data)
                finish_partition = begin_partition
                del partition_data

        del tracks_training
        del tracks_testing

        return experiment_id

    def get_training_data(self, experiment_id):
        """
        Given experiment id, return training data associated with this experiment
        :param experiment_id: int value representing experiment
        :return: dataframe
        """
        return self.experiments[experiment_id]['training']

    def get_testing_data(self, experiment_id):
        """
        Given experiment id, return testing data associated with this experiment
        :param experiment_id: int value representing experiment
        :return: dataframe
        """
        return self.experiments[experiment_id]['testing']

    def get_norm_scalier(self, experiment_id):
        """
        Given experiment id, return normalization scalier, that is testing data length / training data length
        :param experiment_id: int value representing experiment
        :return: float representing normalization scalier
        """
        return self.experiments[experiment_id]['norm_scalier']

    def get_training_partitions(self, experiment_id):
        """
        Given experiment id, return paritioned training data associated with this experiment
        :param experiment_id: int value representing experiment
        :return: list of dataframes or None if data is not partitioned
        """
        try:
            return self.experiments[experiment_id]['training_partitions']
        except KeyError:
            return None
