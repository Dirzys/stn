from evaluation import Evaluator
import pandas as pd
import math


class Model(object):

    def __init__(self, type, data, experiment_id, n_most_often=None):
        self.type = type
        self.training_data = data.get_training_data(experiment_id)
        self.testing_data = data.get_testing_data(experiment_id)
        self.norm_scalier = data.get_norm_scalier(experiment_id)
        self.training_partitions = data.get_training_partitions(experiment_id)
        self.n_most_often = n_most_often
        self.predicted_tracks = None

    def get_unique_user_tracks(self, dataframe):
        return dataframe[['user_id', 'track_id']].drop_duplicates()

    def get_unique_tracks(self, dataframe):
        return dataframe['track_id'].drop_duplicates().values

    def get_unique_users(self, dataframe):
        return dataframe['user_id'].drop_duplicates().values

    def predict_all_previous_tracks(self):
        all_tracks = self.get_unique_tracks(self.training_data)
        all_users = self.get_unique_users(self.training_data)
        return self.to_same_user_track_map(all_tracks, all_users)

    def to_same_user_track_map(self, predicted_tracks, users):
        user_track_map = {}
        for user in users:
            user_track_map[user] = predicted_tracks
        return user_track_map

    def predict_user_previous_tracks(self):
        return self.to_user_track_map(self.get_unique_user_tracks(self.training_data).values)

    def to_user_track_map(self, predicted_track_dataframe):
        user_track_map = {}
        for user, listened_track in predicted_track_dataframe:
            if user in user_track_map:
                user_track_map[user].append(listened_track)
            else:
                user_track_map[user] = [listened_track]
        return user_track_map

    def on_average_user_listened_songs(self):
        if self.training_partitions is None:
            raise Exception("Training data is not partitioned!")
        user_unique_track_counts = pd.concat([self.get_unique_user_tracks(partition)
                                              for partition in self.training_partitions]).groupby('user_id').size().reset_index()

        user_unique_track_counts[0] = user_unique_track_counts[0]*self.norm_scalier
        return user_unique_track_counts

    def select_n_most_frequent(self, dataframe, n):
        return dataframe.groupby(['user_id', 'track_id']).agg({'track_name': 'count'})\
            .rename(columns={'track_name': 'count'})['count']\
            .groupby(level=0, group_keys=False).nlargest(n).reset_index()

    def predict_average_most_often(self):
        merged = pd.merge(self.training_data, self.on_average_user_listened_songs(), on='user_id')

        def select_n_most_frequent(group):
            key = int(math.ceil(group.iloc[0][0]))
            return self.select_n_most_frequent(group, key)

        return merged.groupby([0]).apply(select_n_most_frequent)

    def predict_n_most_often(self):
        if self.n_most_often is None:
            raise Exception("The number of most frequently user listened songs is missing.")
        if self.n_most_often == 0:
            prediction = self.predict_average_most_often()
        else:
            prediction = self.select_n_most_frequent(self.training_data, self.n_most_often)
        return self.to_user_track_map(prediction[['user_id', 'track_id']].values)

    def predict(self):
        self.predicted_tracks = {
            'all_previous_tracks': self.predict_all_previous_tracks,
            'user_previous_tracks': self.predict_user_previous_tracks,
            'n_most_often': self.predict_n_most_often
        }[self.type]()

    def evaluate(self, name, pprint):
        if name is None:
            name = self.type
        evaluation = Evaluator()
        score = evaluation.score(self.predicted_tracks,
                                 self.to_user_track_map(self.get_unique_user_tracks(self.testing_data).values))
        if pprint:
            evaluation.pprint_scores(score, name)
        return score

    def run(self, name=None, pprint=True):
        self.predict()
        score = self.evaluate(name, pprint)
        return score




