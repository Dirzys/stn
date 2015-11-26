from evaluation import Evaluator
import pandas as pd


class Model(object):
    """ Model class is responsible for running and evaluating models """
    def __init__(self, data, experiment_id, n_most_often=None, cluster=None, from_clusters='top'):
        """
        Initialise model with the following parameters:
        :param data: Data object
        :param experiment_id: int, experiment id
        :param n_most_often: int representing how many most often listened tracks to return
                (required for NMostOften model)
        :param cluster: TrackClustering object (required for models that select tracks from track networks)
        :param from_clusters: 'top' or 'all (required for NFromCluster model)
        """
        self.training_data = data.get_training_data(experiment_id)
        self.testing_data = data.get_testing_data(experiment_id)
        self.norm_scalier = data.get_norm_scalier(experiment_id)
        self.training_partitions = data.get_training_partitions(experiment_id)
        self.n_most_often = n_most_often
        self.cluster = cluster
        self.from_clusters = from_clusters
        self.type = self.__class__.__name__
        self.predicted_tracks = None

    def get_unique_user_tracks(self, dataframe):
        """
        Given dataframe find unique tracks for each user
        :return: dataframe consisting columns ('user_id', 'track_id')
        """
        return dataframe[['user_id', 'track_id']].drop_duplicates()

    def get_unique_tracks(self, dataframe):
        """
        Find unique tracks in the given dataframe
        :return: numpy array of unique track IDs
        """
        return dataframe['track_id'].drop_duplicates().values

    def get_unique_users(self, dataframe):
        """
        Find unique users in the given dataframe
        :return: numpy array of unique user IDs
        """
        return dataframe['user_id'].drop_duplicates().values

    def to_user_track_map(self, track_dataframe):
        """
        Given track dataframe return user track map that can be used to evaluate predictions
        :param track_dataframe: dataframe consisting only two columns ('user_id', 'track_id')
        :return: user track map: {user_id, [track_id_1, track_id_2, ...], ...}
        """
        user_track_map = {}
        for user, listened_track in track_dataframe:
            if user in user_track_map:
                user_track_map[user].append(listened_track)
            else:
                user_track_map[user] = [listened_track]
        return user_track_map

    def predict(self):
        raise NotImplementedError

    def evaluate(self, name, pprint):
        """
        Evaluate model using predicted tracks and actually listened tracks.
        :param name: string, name of the experiment (if None, class name will be printed instead)
        :param pprint: bool, if True -> scores will be pretty printed
        :return: score of the model as a tuple (precision, recall, f-score)
        """
        if name is None:
            name = self.type
        evaluation = Evaluator()
        score = evaluation.score(self.predicted_tracks,
                                 self.to_user_track_map(self.get_unique_user_tracks(self.testing_data).values))
        if pprint:
            evaluation.pprint_scores(score, name)
        return score

    def run(self, name=None, pprint=True):
        """
        Run the model.
        :param name: string, name of the experiment (if None, class name will be printed instead)
        :param pprint: bool, if True -> scores will be pretty printed
        :return: score of the model as a tuple (precision, recall, f-score)
        """
        self.predict()
        score = self.evaluate(name, pprint)
        return score


class AllPreviousTracks(Model):
    """ Simple model that predicts all unique tracks in the training data for each user """
    def predict(self):
        all_tracks = self.get_unique_tracks(self.training_data)
        all_users = self.get_unique_users(self.training_data)
        self.predicted_tracks = self.to_same_user_track_map(all_tracks, all_users)

    def to_same_user_track_map(self, predicted_tracks, users):
        """
        Given the list of tracks and list of users create a user track map {user_id: predicted_tracks, ...}
        :param predicted_tracks: list of tracks
        :param users: list of users
        :return: user track map: {user_id: predicted_tracks, ...}
        """
        user_track_map = {}
        for user in users:
            user_track_map[user] = predicted_tracks
        return user_track_map


class UserPreviousTracks(Model):
    """ Model that predicts all unique tracks that user has played already """
    def predict(self):
        self.predicted_tracks = self.to_user_track_map(self.get_unique_user_tracks(self.training_data).values)


class NMostOften(Model):
    """
    NMostOften model returns N most often played tracks by each user where N is specified
    during model initialisation. If N=0, the model will select a different N for each user,
    equal to unique tracks user listened per self.testing_length days on average.
    """
    def predict(self):
        if self.n_most_often is None:
            raise Exception("The number of most frequently user listened tracks is missing.")
        if self.n_most_often == 0:
            prediction = self.predict_average_most_often()
        else:
            prediction = self.select_n_most_often(self.training_data, self.n_most_often)
        self.predicted_tracks = self.to_user_track_map(prediction[['user_id', 'track_id']].values)

    def select_n_most_often(self, dataframe, n):
        """
        Given a dataframe consisting user listened track IDs and names,
        return n most often played tracks for each user
        :return: dataframe with the columns ('user_id', 'track_id', 'count')
        """
        return dataframe.groupby(['user_id', 'track_id']).agg({'track_name': 'count'})\
            .rename(columns={'track_name': 'count'})['count']\
            .groupby(level=0, group_keys=False).nlargest(n).reset_index()

    def predict_average_most_often(self):
        """
        Find N_u for each user and return N_u most often listened tracks for each user
        :return: dataframe with the columns ('user_id', 'track_id', '0') ('0' represents count)
        """
        merged = pd.merge(self.training_data, self.on_average_user_listened_songs(), on='user_id')

        def select_n_most_frequent(group):
            key = int(round(group.iloc[0][0]))
            return self.select_n_most_often(group, key)

        return merged.groupby([0]).apply(select_n_most_frequent)

    def on_average_user_listened_songs(self):
        """
        Given training data set partitions, find the average number of
        unique tracks user has listened per self.testing_length days interval
        :return: dataframe with the columns ('user_id', '0'), where '0' column represents
                the average number of tracks listened per self.testing_length days
        """
        if self.training_partitions is None:
            raise Exception("Training data is not partitioned!")
        user_unique_track_counts = pd.concat([self.get_unique_user_tracks(partition)
                                              for partition in self.training_partitions]).groupby('user_id').size().reset_index()

        user_unique_track_counts[0] = user_unique_track_counts[0]*self.norm_scalier
        return user_unique_track_counts


class NFromCluster(Model):
    def predict(self):
        user_cluster_frequency = self.user_cluster_frequency()
        if self.from_clusters == 'all':
            self.predicted_tracks = self.get_all_tracks_from_clusters(user_cluster_frequency)
        if self.from_clusters == 'top':
            prediction = self.get_tracks_from_clusters_by_user_track_freq(user_cluster_frequency)
            self.predicted_tracks = self.to_user_track_map(prediction[['user_id', 'track_id']].values)

    def map_to_cluster(self, track_id):
        """
        Return cluster ID assigned for the given track
        """
        return self.cluster.tracks_map[track_id]['cluster_id']

    def get_all_tracks_from_clusters(self, user_cluster_frequency):
        """
        Given user cluster frequency dataframe, for each user return all unique tracks from all clusters
        that are listened by user at least 0.5 times on average per self.testing_length days period.
        :param user_cluster_frequency: dataframe consisting columns ('user_id', 'cluster_id')
        :return: user_track_map that can be used for evaluation - {user_id: [track_id_1, track_id_2, ...], ...}
        """
        user_track_map = {}
        for user_cluster in user_cluster_frequency[['user_id', 'cluster_id']].drop_duplicates().values:
            user_id, cluster_id = user_cluster
            if user_id not in user_track_map:
                user_track_map[user_id] = []
            user_track_map[user_id].extend(self.cluster.clusters[cluster_id])
        return user_track_map

    def get_tracks_from_clusters_by_user_track_freq(self, user_cluster_frequency):
        """
        Given user cluster frequency dataframe for each user sort (in the decreasing order)
        all tracks in the clusters w.r.t how many times user has played each track
        and return only the top N, where N is given in count column of user_cluster_frequency dataframe
        :param user_cluster_frequency: dataframe consisting columns ('user_id', 'cluster_id', '0'), where '0' columns
                represents the average number of times user has played track from cluster per self.testing_length
                days interval
        """
        user_unique_track_counts = self.training_data.groupby(['user_id', 'track_id']).agg({'track_name': 'count'})\
            .rename(columns={'track_name': 'count'}).reset_index()

        user_unique_track_counts['cluster_id'] = user_unique_track_counts['track_id'].apply(self.map_to_cluster)

        user_cluster_frequency_map = {}
        for user_cluster in user_cluster_frequency.values:
            user_id, cluster_id, count = user_cluster
            if user_id not in user_cluster_frequency_map:
                user_cluster_frequency_map[user_id] = {}
            user_cluster_frequency_map[user_id][cluster_id] = int(round(count))

        def most_often_from_cluster(group):
            cluster_id = group.iloc[0]['cluster_id']
            user_id = group.iloc[0]['user_id']
            count = user_cluster_frequency_map[user_id][cluster_id]
            if count == 0:
                return
            else:
                return group.sort('count', ascending=False).head(int(round(count)))

        return user_unique_track_counts.groupby(['user_id', 'cluster_id']).apply(most_often_from_cluster)

    def user_cluster_frequency(self):
        """
        Given partitioned training data, for each user and each cluster find
        the average number of times user has played the track from the cluster per self.testing_length days interval
        :return: dataframe with the columns ('user_id', 'cluster_id', '0'), where '0' represents the average count
        """
        if self.training_partitions is None:
            raise Exception("Training data is not partitioned!")
        user_unique_tracks = pd.concat([self.get_unique_user_tracks(partition) for partition in self.training_partitions])

        user_unique_tracks['cluster_id'] = user_unique_tracks['track_id'].apply(self.map_to_cluster)
        user_tracks_from_cluster_counts = user_unique_tracks.groupby(['user_id', 'cluster_id']).size().reset_index()
        user_tracks_from_cluster_counts[0] = user_tracks_from_cluster_counts[0]*self.norm_scalier
        user_tracks_from_cluster_counts = user_tracks_from_cluster_counts[user_tracks_from_cluster_counts[0] >= 0.5]
        return user_tracks_from_cluster_counts


class CommonNeighborsWithinCluster(NFromCluster):
    """
    Model that predicts tracks for the user by selecting tracks from user clusters that
    have at least two similar tracks (direct edge in the track network) that have been played by the user in the past
    """
    def predict(self):
        user_cluster_frequency = self.user_cluster_frequency()
        predicted_by_freq = self.get_tracks_from_clusters_by_user_track_freq(user_cluster_frequency)
        self.predicted_tracks = self.predict_tracks(predicted_by_freq[['track_id']].reset_index()[['user_id', 'cluster_id', 'track_id']])

    def predict_tracks(self, tracks, from_cluster=True):
        """
        Given a dataframe of user listened tracks and their corresponding cluster IDs,
        select tracks that have at least two similar tracks in the same cluster (direct edge in the track network)
        that have been played by the user in the past
        :param tracks: dataframe consisting columns ('user_id', 'cluster_id', 'track_id')
        :return: predicted track map for each user: {user_id: [track_id, ...], ...}
        """
        tracks['track_int_id'] = tracks['track_id'].apply(self.map_to_track_int_id)

        on = ['user_id', 'cluster_id']
        if not from_cluster:
            on = ['user_id']
        merged = pd.merge(tracks, tracks, on=on)
        merged = merged[merged['track_int_id_x'] != merged['track_int_id_y']]

        all_neighbors = {}

        _ = merged.apply(self.get_common_neighbors, args=(all_neighbors,), axis=1)

        for user in all_neighbors:
            unique_neighbors = list(set(all_neighbors[user]))
            all_neighbors[user] = [self.map_to_track_id(neighbor) for neighbor in unique_neighbors]

        return all_neighbors

    def get_common_neighbors(self, two_tracks_in_cluster, all_neighbors):
        """
        Given dataframe row consisting two track integer IDs from the same cluster,
        find all shared neighbors in the song network
        :param two_tracks_in_cluster: dataframe row consisting columns ('track_in_id_x', 'track_in_id_y', 'user_id')
        :param all_neighbors: dictionary at which found neighbors for user are stored
        """
        track_int_id_1 = two_tracks_in_cluster['track_int_id_x']
        track_int_id_2 = two_tracks_in_cluster['track_int_id_y']
        user_id = two_tracks_in_cluster['user_id']
        neighbors = self.cluster.get_common_neighbors(track_int_id_1, track_int_id_2)
        if user_id not in all_neighbors:
            all_neighbors[user_id] = []
        all_neighbors[user_id].extend(neighbors)

    def map_to_track_int_id(self, track_id):
        """
        Map track ID into track integer ID
        """
        return self.cluster.tracks_map[track_id]['int_id']

    def map_to_track_id(self, track_int_id):
        """
        Map from track integer ID to track ID
        """
        return self.cluster.labels[track_int_id]


class CommonNeighbors(CommonNeighborsWithinCluster, NMostOften):
    """
    Model that predicts tracks for the user by selecting tracks that have at least two similar tracks
    (direct edge in the track network) that have been played by the user in the past.
    Model is similar to CommonNeighborsWithinCluster, however it also considers track neighbors that
    are outside of the cluster.
    Note: Not recommended to use on the big datasets as it is very slow
    """
    def predict(self):
        self.predicted_tracks = self.predict_tracks(self.predict_average_most_often()[['user_id', 'track_id']], from_cluster=False)
