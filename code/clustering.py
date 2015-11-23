import pandas as pd
import numpy
import matplotlib.pyplot as plt
import snap


class TrackClustering(object):

    def __init__(self, training_data, maximum_diff, minimum_similarity=None):
        self.training_data = training_data
        self.maximum_diff = maximum_diff
        self.minimum_similarity = minimum_similarity
        self.tracks_map = {}
        self.cons_listened_counts = None
        self.song_network = None
        self.labels = None
        self.clusters = {}

    def prepare(self):
        # Helper dataframe where all listened songs are shifted once so that it can be concatenated
        # with the original all_listened dataframe. The resulting dataframe holds rows for all
        # consecutively listen tracks
        training_helper = self.training_data.rename(
            columns={'user_id': 'user_id_2',
                     'timestamp': 'timestamp_2',
                     'track_id': 'track_id_2',
                     'track_name': 'track_name_2'}).shift()
        cons_listened = pd.concat([self.training_data, training_helper], axis=1)

        # Remove rows for which user id is not the same, tracks are identical
        # or the time difference between two consecutive tracks is bigger than the thredhsold MIN
        cons_listened = cons_listened[(cons_listened['user_id'] == cons_listened['user_id_2']) &
                                      (cons_listened['track_id'] != cons_listened['track_id_2']) &
                                      ((cons_listened['timestamp_2']-cons_listened['timestamp'])
                                       .astype('timedelta64[m]') <= self.maximum_diff)]

        # Count how many times each two tracks were played one after other in MIN minutes period
        self.cons_listened_counts = pd.DataFrame({'cons_count': cons_listened.groupby(
            numpy.vectorize(self.cons_group)(cons_listened['track_id'], cons_listened['track_id_2'])
        ).size()}).reset_index()

        # Count how many times each song were played in total
        track_counts = self.training_data.groupby(['track_id', 'track_name']).size()

        # Create a tracks map where each track id -> track name, track total count, integer id
        for i, (track_id, track_name) in enumerate(track_counts.keys()):
            self.tracks_map[track_id] = {'track_name': track_name,
                                    'track_count': track_counts[(track_id, track_name)],
                                    'int_id': i}

        # Add similarity column that determines the similarity
        # between two tracks in the row calculated using similarity function
        self.cons_listened_counts['similarity'] = numpy.vectorize(self.similarity)(self.cons_listened_counts['cons_count'],
                                                                                   self.cons_listened_counts['level_0'],
                                                                                   self.cons_listened_counts['level_1'])

    def cons_group(self, track_1_id, track_2_id):
        """ Function to determine the group based on two tracks presented in the row. """
        if track_1_id > track_2_id:
            return track_1_id, track_2_id
        else:
            return track_2_id, track_1_id

    def similarity(self, cons_count, track_1_count, track_2_count):
        """ Function to calculate the similarity between two songs
            given both tracks total counts in the corpus and the
            number of times both tracks were played one after other"""
        return cons_count*1.0/numpy.sqrt(self.tracks_map[track_1_count]['track_count'] *
                                         self.tracks_map[track_2_count]['track_count'])

    def plot_similarities(self):
        # Plot the similarities between songs in increasing order
        plt.plot(self.cons_listened_counts['similarity'].sort(inplace=False))

    def create_network(self, minimum_similarity=None):
        if minimum_similarity is None:
            if self.minimum_similarity is None:
                raise Exception("Minimum similarity is not specified")
            else:
                minimum_similarity = self.minimum_similarity
        # Create song network, add all tracks as nodes in it
        self.song_network = snap.TUNGraph.New()
        self.labels = {}
        for track_id in self.tracks_map:
            int_id = self.tracks_map[track_id]['int_id']
            self.song_network.AddNode(int_id)
            self.labels[int_id] = track_id

        # Only keep the similar tracks and add edges between these tracks in the network specified above
        similar_songs = self.cons_listened_counts[self.cons_listened_counts['similarity'] > minimum_similarity].values

        for similar in similar_songs:
            self.song_network.AddEdge(self.tracks_map[similar[0]]['int_id'], self.tracks_map[similar[1]]['int_id'])

        # print song_network.GetEdges()

    def cluster(self):
        # Cluster the tracks network using Clauset-Newman-Moore community detection method
        network_clusters = snap.TCnComV()
        modularity = snap.CommunityCNM(self.song_network, network_clusters)
        print "The modularity of the network is %f" % modularity

        # Append cluster ids to all tracks in tracks map
        for i, cluster in enumerate(network_clusters):
            self.clusters[i] = []
            for int_id in cluster:
                track_id = self.labels[int_id]
                self.tracks_map[track_id]['cluster_id'] = i
                self.clusters[i].append(track_id)

    def run(self, plot_similarities=False):
        self.prepare()
        if plot_similarities:
            self.plot_similarities()
        self.create_network()
        self.cluster()

    def get_common_neighbors(self, node_id_1, node_id_2):
        Nbrs = snap.TIntV()
        n = snap.GetCmnNbrs(self.song_network, node_id_1, node_id_2, Nbrs)
        return list(Nbrs)
