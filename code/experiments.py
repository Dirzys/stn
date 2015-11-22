from code.models import Model
from datetime import timedelta
import matplotlib.pyplot as plt


class Experiment(object):
    def __init__(self, training_lengths, testing_lengths, testing_finish_dates):
        self.training_lengths = training_lengths
        self.testing_lengths = testing_lengths
        self.testing_finish_dates = testing_finish_dates
        self.scores = []

    def run(self, data, exp_id, as_graph=True):
        print "==== Model user_previous_tracks ===="
        for training_length in self.training_lengths:
            for testing_length in self.testing_lengths:
                for finish_testing in self.testing_finish_dates:
                    _ = data.create_experiment_data(training_length, testing_length, finish_testing, exp_id)
                    model = Model('user_previous_tracks', data, exp_id)
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
        return {0: "The length of training data (days)",
                1: "The length of testing data (days)",
                2: "Testing data finish date"}[param_id]


class NMostOftenExperiment(object):
    def __init__(self, training_length, testing_length, finish_testing, data, exp_id, n_values):
        _ = data.create_experiment_data(training_length, testing_length, finish_testing, exp_id, True)
        self.training_length = training_length
        self.testing_length = testing_length
        self.finish_testing = finish_testing
        self.n_values = n_values
        self.data = data
        self.exp_id = exp_id

    def run(self):
        scores = []
        score_for_average = None
        n_values = self.n_values[1:] if self.n_values[0] == 0 else self.n_values
        for n in self.n_values:
            model = Model('n_most_often', self.data, self.exp_id, n)
            score = model.run(pprint=False)
            if n == 0:
                score_for_average = score
            else:
                scores.append(score)

        precisions, recalls, f_scores = zip(*scores)
        plt.plot(n_values, precisions, 'b-', label="Precision")
        plt.plot(n_values, recalls, 'g-', label="Recall")
        plt.plot(n_values, f_scores, 'r-', label="F-score")
        xmin = 0
        xmax = 1
        plt.axhline(score_for_average[0], xmin=xmin, xmax=xmax, color="b", linestyle="--")
        plt.axhline(score_for_average[1], xmin=xmin, xmax=xmax, color="g", linestyle="--")
        plt.axhline(score_for_average[2], xmin=xmin, xmax=xmax, color="r", linestyle="--")
        plt.title("Precision, recall and f-score when training data consists %s days and \n "
                  "testing data consists %s days of all users track behaviour \n"
                  "and testing data finish date is %s" %
                  (self.training_length, self.testing_length, self.finish_testing.strftime("%Y-%m-%d")))
        plt.xlabel("The number of most frequently listened tracks for each user to be predicted")
        plt.legend(loc='best')
        plt.show()


