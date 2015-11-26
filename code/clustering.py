import pandas as pd
import numpy
import matplotlib.pyplot as plt
import snap


class TrackClustering(object):
    """
    TrackClustering class that is responsible for track network creation,
    track clustering and any other operations on the network.
    """
    def __init__(self, training_data, maximum_diff, minimum_similarity=None):
        """
        Initialise the clustering object by providing training data, maximum allowed
        time difference between two tracks to be included into similarity calculation (m hat) and
        minimum similarity between two tracks to be considered as similar (s hat).
        :param training_data: dataframe consisting training data
        :param maximum_diff: int, minutes
        :param minimum_similarity: float
        """
        self.training_data = training_data
        self.maximum_diff = maximum_diff
        self.minimum_similarity = minimum_similarity
        self.tracks_map = {}
        self.cons_listened_counts = None
        self.song_network = None
        self.labels = None
        self.clusters = {}

    def prepare(self):
        """
        Prepare data for clustering by:
         1. Counting how many times each pair of tracks were played one after the other within self.maximum_diff period
         2. Counting how many times each track were played in total
         3. Creating a track map: {track_id: {'track_name': track_name, 'track_count': track_total_count,
                                            'int_id': integer_id}, ...}
         4. Calculating the similarity between each pair of tracks
        """
        # Helper dataframe where all listened songs are shifted once so that it can be concatenated
        # with the original training data dataframe. The resulting dataframe holds rows for all
        # consecutively listen tracks
        training_helper = self.training_data.rename(
            columns={'user_id': 'user_id_2',
                     'timestamp': 'timestamp_2',
                     'track_id': 'track_id_2',
                     'track_name': 'track_name_2'}).shift()
        cons_listened = pd.concat([self.training_data, training_helper], axis=1)

        # Remove rows for which user id is not the same, tracks are identical
        # or the time difference between two consecutive tracks is bigger than the threshold self.maximum_diff
        cons_listened = cons_listened[(cons_listened['user_id'] == cons_listened['user_id_2']) &
                                      (cons_listened['track_id'] != cons_listened['track_id_2']) &
                                      ((cons_listened['timestamp_2']-cons_listened['timestamp'])
                                       .astype('timedelta64[m]') <= self.maximum_diff)]

        # Count how many times each two tracks were played one after other in self.maximum_diff period
        self.cons_listened_counts = pd.DataFrame({'cons_count': cons_listened.groupby(
            numpy.vectorize(self.cons_group)(cons_listened['track_id'], cons_listened['track_id_2'])
        ).size()}).reset_index()

        # Count how many times each track were played in total
        track_counts = self.training_data.groupby(['track_id', 'track_name']).size()

        # Create a track map where each track id -> track name, track total count, integer id
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
        """
        Function to calculate the similarity between two songs
        given both tracks total counts in the corpus and the
        number of times both tracks were played one after the other.
        Similarity is determined by cosine similarity measure s(t1, t2) = count(t1, t2)/sqrt(count(t1), count(t2))
        """
        return cons_count*1.0/numpy.sqrt(self.tracks_map[track_1_count]['track_count'] *
                                         self.tracks_map[track_2_count]['track_count'])

    def plot_similarities(self):
        """ Plot the similarities between pairs of tracks in increasing order """
        plt.plot(self.cons_listened_counts['similarity'].sort(inplace=False))

    def create_network(self, minimum_similarity=None):
        """
        Create track network by adding all tracks as nodes and adding links between two nodes only if
         the corresponding tracks are similar (similarity higher than minimum_similarity)
        """
        if minimum_similarity is None:
            if self.minimum_similarity is None:
                raise Exception("Minimum similarity is not specified")
            else:
                minimum_similarity = self.minimum_similarity
        # Create track network, add all tracks as nodes in it
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

    def cluster(self):
        """
        Run Clauset-Newman-Moore community detection algorithm on self.song_network,
        print the modularity of the network,
        append corresponding cluster IDs to all tracks in tracks map self.tracks_map
        and save clusters as {cluster_id: [track_id,...],...} into self.clusters map.
        """
        network_clusters = snap.TCnComV()
        modularity = snap.CommunityCNM(self.song_network, network_clusters)
        print "The modularity of the network is %f" % modularity

        for i, cluster in enumerate(network_clusters):
            self.clusters[i] = []
            for int_id in cluster:
                track_id = self.labels[int_id]
                self.tracks_map[track_id]['cluster_id'] = i
                self.clusters[i].append(track_id)

    def run(self, plot_similarities=False):
        """
        Prepare data for tracks network creation,
        create track network and cluster it.
        :param plot_similarities: If true, plot similarities between all pairs of tracks in increasing order to
        inspect the reliable minimum similarity measure.
        """
        self.prepare()
        if plot_similarities:
            self.plot_similarities()
        self.create_network()
        self.cluster()

    def get_common_neighbors(self, node_id_1, node_id_2):
        """
        Get common neighbors between two nodes.
        :param node_id_1: Track 1 integer id
        :param node_id_2: Track 2 integer id
        :return: list of nodes (track integer ids) that has a direct link with both node_id_1 and node_id_2
        """
        Nbrs = snap.TIntV()
        _ = snap.GetCmnNbrs(self.song_network, node_id_1, node_id_2, Nbrs)
        return list(Nbrs)
