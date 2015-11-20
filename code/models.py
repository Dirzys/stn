from evaluation import Evaluator


class Model(object):

    def __init__(self, type, data, experiment_id):
        self.type = type
        self.training_data = data.get_training_data(experiment_id)
        self.testing_data = data.get_testing_data(experiment_id)
        self.predicted_tracks = None

    def get_unique_user_tracks(self, dataframe):
        return dataframe[['user_id', 'track_id']].drop_duplicates().values

    def predict_user_previous_tracks(self):
        return self.to_user_track_map(self.get_unique_user_tracks(self.training_data))

    def to_user_track_map(self, predicted_track_dataframe):
        user_track_map = {}
        for user, listened_track in predicted_track_dataframe:
            if user in user_track_map:
                user_track_map[user].append(listened_track)
            else:
                user_track_map[user] = [listened_track]
        return user_track_map

    def predict(self):
        self.predicted_tracks = {
            'user_previous_tracks': self.predict_user_previous_tracks
        }[self.type]()

    def evaluate(self, name, pprint):
        if name is None:
            name = self.type
        evaluation = Evaluator()
        score = evaluation.score(self.predicted_tracks,
                                 self.to_user_track_map(self.get_unique_user_tracks(self.testing_data)))
        if pprint:
            evaluation.pprint_scores(score, name)
        return score

    def run(self, name=None, pprint=True):
        self.predict()
        score = self.evaluate(name, pprint)
        return score




