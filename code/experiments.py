from datetime import timedelta
import matplotlib.pyplot as plt
from models import *
from code.clustering import TrackClustering


class DataExperiment(object):
    """
    DataExperiment class tries different training and testing data slices for given model type
    """
    def __init__(self, training_lengths, testing_lengths, testing_finish_dates, model):
        """
        Initialise the experiment by providing parameter values for training, testing data lengths,
        when the testing data finishes and model type
        :param training_lengths: list of int representing training data length in days
        :param testing_lengths: list of int representing testing data length in days
        :param testing_finish_dates: list of datetime objects representing when the testing data finishes
        :param model: model class to be ran
        """
        self.training_lengths = training_lengths
        self.testing_lengths = testing_lengths
        self.testing_finish_dates = testing_finish_dates
        self.model = model
        self.scores = []

    def run(self, data, exp_id, as_graph=True):
        """
        Run the experiment by running given model with all possible combinations of parameter values.
        Plot the graphs of precision, recall and f-score if required. Otherwise, pprint the scores.
        :param data: Data object
        :param exp_id: int representing experiment
        :param as_graph: bool
        """
        for training_length in self.training_lengths:
            for testing_length in self.testing_lengths:
                for finish_testing in self.testing_finish_dates:
                    _ = data.create_experiment_data(training_length, testing_length, finish_testing, exp_id)
                    model = self.model(data, exp_id)
                    begin_testing = finish_testing - timedelta(days=testing_length)
                    begin_training = (begin_testing - timedelta(days=training_length)).strftime("%Y-%m-%d")
                    score = model.run("Training from %s (%s days), testing from %s (%s days)" %
                                      (begin_training, training_length, begin_testing.strftime("%Y-%m-%d"), testing_length),
                                      pprint=(not as_graph))
                    if as_graph:
                        self.scores.append(((training_length, testing_length, finish_testing), score))
        if as_graph:
            self.draw_graphs()

    def draw_graphs(self):
        """
        Draw the graphs for the experiment. Plot as few graphs as possible.
        """
        params = {0: self.training_lengths,
                  1: self.testing_lengths,
                  2: self.testing_finish_dates}
        longest, second, shortest = sorted(params.items(), key=lambda x: len(x[1]), reverse=True)

        for param_i in shortest[1]:
            for param_j in second[1]:
                key = [None]*3
                key[shortest[0]] = param_i
                key[second[0]] = param_j
                scores = []
                for score_params, score in self.scores:
                    if (score_params[0] == key[0] or key[0] is None) and (score_params[1] == key[1] or key[1] is None) and (score_params[2] == key[2] or key[2] is None):
                        scores.append(score)
                precisions, recalls, f_scores = zip(*scores)
                plt.plot(longest[1], precisions, 'b-', label="Precision")
                plt.plot(longest[1], recalls, 'g-', label="Recall")
                plt.plot(longest[1], f_scores, 'r-', label="F-score")
                plt.title(self.create_title(second[0], shortest[0], param_j, param_i))
                plt.legend(loc='best')
                plt.xlabel(self.create_x_label(longest[0]))
                plt.show()

    def create_title(self, param_id_1, param_id_2, param_1, param_2):
        """
        Create title for the graph given two frozen parameters and thei values.
        :param param_id_1: Frozen parameter (0: training data param, 1: testing data param, 2: testing data finish date param)
        :param param_id_2: Frozen parameter (param_id_1 != param_id_2)
        :param param_1: Parameter 1 value
        :param param_2: Parameter 2 value
        :return: string representing the title of the graph
        """
        if param_id_1 > param_id_2:
            tmp = param_id_2
            param_id_2 = param_id_1
            param_id_1 = tmp
            tmp = param_2
            param_2 = param_1
            param_1 = tmp
        if param_id_2 == 2:
            param_2 = param_2.strftime("%Y-%m-%d")
        return {"01": "Precision, recall and f-score when training data consists %s days and \n"
                      "testing data consists %s days of all users track behaviour" % (param_1, param_2),
                "02": "Precision, recall and f-score when training data consists %s days of all users \n"
                      "track behaviour and testing data finish date is %s" % (param_1, param_2),
                "12": "Precision, recall and f-score when testing data consists %s days of all users \n"
                      "track behaviour and testing data finish date is %s" % (param_1, param_2)
                }["%s%s" % (param_id_1, param_id_2)]

    def create_x_label(self, param_id):
        """
        Create x label for the plot given free parameter
        :param param_id: Free parameter (0: training data, 1: testing data, 2: testing data finish date)
        :return: string representing the x label of the graph
        """
        return {0: "The length of training data (days)",
                1: "The length of testing data (days)",
                2: "Testing data finish date"}[param_id]


class NMostOftenExperiment(object):
    """
    NMostOftenExperiment runs n_most_often model with different values of N (how many most often
    listened tracks to select for each user)
    """
    def __init__(self, training_length, testing_length, finish_testing, data, exp_id, n_values):
        """
        Initialise the experiment by providing parameter values for training, testing data lengths,
        when the testing data finishes, data source, experiment id and different N values to be tried.
        Create the experiment data.
        :param training_length: int representing training data length in days
        :param testing_length: int representing testing data length in days
        :param finish_testing: datetime object representing when the testing data finishes
        :param data: Data object
        :param exp_id: experiment id
        :param n_values: list of int representing N values to be tried
        :return:
        """
        _ = data.create_experiment_data(training_length, testing_length, finish_testing, exp_id, True)
        self.training_length = training_length
        self.testing_length = testing_length
        self.finish_testing = finish_testing
        self.n_values = n_values
        self.data = data
        self.exp_id = exp_id
        self.scores = []
        self.score_for_average = None

    def run(self):
        """
        Run n_most_often model for each given N value. If N=0, the model will select a different N for each user,
        equal to unique tracks user listened per self.testing_length days on average.
        At the end the graph is plotted.
        """
        for n in self.n_values:
            model = NMostOften(self.data, self.exp_id, n_most_often=n)
            score = model.run(pprint=False)
            if n == 0:
                self.score_for_average = score
            else:
                self.scores.append(score)

        self.draw_graph()

    def draw_graph(self):
        """
        Draw a graph that plots the precision, recall and f-score curves for n_most_often model with
        each N provided for an experiment.
        If there were N=0, plot three dotted lines representing what the metrics are if N is different for each user
         (in fact, equal to unique tracks user listened per self.testing_length days on average)
        """
        n_values = self.n_values[1:] if self.n_values[0] == 0 else self.n_values
        precisions, recalls, f_scores = zip(*self.scores)
        plt.plot(n_values, precisions, 'b-', label="Precision")
        plt.plot(n_values, recalls, 'g-', label="Recall")
        plt.plot(n_values, f_scores, 'r-', label="F-score")
        average_string = ""
        if self.score_for_average is not None:
            xmin = 0
            xmax = 1
            plt.axhline(self.score_for_average[0], xmin=xmin, xmax=xmax, color="b", linestyle="--")
            plt.axhline(self.score_for_average[1], xmin=xmin, xmax=xmax, color="g", linestyle="--")
            plt.axhline(self.score_for_average[2], xmin=xmin, xmax=xmax, color="r", linestyle="--")
            average_string = ".\nDotted lines represent the metrics if the number of predicted tracks for each \n" \
                             "user is equal to unique tracks user listened per %s days on average" % self.testing_length
        plt.title("Precision, recall and f-score when training data consists %s days and testing data \n"
                  "consists %s days of all users track behaviour and testing data finish date is %s%s" %
                  (self.training_length, self.testing_length, self.finish_testing.strftime("%Y-%m-%d"), average_string))
        plt.xlabel("The number of most frequently listened tracks for each user to be predicted")
        plt.legend(loc='best')
        plt.show()


class ClusteringExperiment(NMostOftenExperiment):
    """
    ClusteringExperiment runs both NFromCluster ('all' and 'top') and CommonNeighborsWithinCluster models with
    given maximum allowed time difference between two tracks to be included into similarity calculation (mins) value
    and draws a separate graph for each model.
    """
    def run(self):
        self.scores = [[], [], []]
        models = ['NFromCluster-All', 'NFromCluster-Top', 'CommonNeighborsWithinCluster']
        for n in self.n_values:
            clustering = TrackClustering(self.data.get_training_data(self.exp_id), 60, n)
            clustering.run()

            model = NFromCluster(self.data, self.exp_id, cluster=clustering, from_clusters='all')
            score = model.run(pprint=False)
            self.scores[0].append(score)

            model = NFromCluster(self.data, self.exp_id, cluster=clustering, from_clusters='top')
            score = model.run(pprint=False)
            self.scores[1].append(score)

            model = CommonNeighborsWithinCluster(self.data, self.exp_id, cluster=clustering)
            score = model.run(pprint=False)
            self.scores[2].append(score)

        self.draw_graph(models)

    def draw_graph(self, models):
        for i, scores in enumerate(self.scores):
            precisions, recalls, f_scores = zip(*scores)
            plt.plot(self.n_values, precisions, 'b-', label="Precision")
            plt.plot(self.n_values, recalls, 'g-', label="Recall")
            plt.plot(self.n_values, f_scores, 'r-', label="F-score")
            plt.title("Precision, recall and f-score for %s model when training data consists %s days and \n"
                      "testing data consists %s days of all users track behaviour and testing data finish date is %s" %
                      (models[i], self.training_length, self.testing_length, self.finish_testing.strftime("%Y-%m-%d")))
            plt.xlabel("Maximum allowed time difference between two tracks "
                       "to be included into similarity calculation (mins)")
            plt.legend(loc='best')
            plt.show()



