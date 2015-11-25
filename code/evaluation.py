import numpy


class Evaluator(object):
    """
    Evaluation class that scores predicted tracks against true tracks and pprints the scores
    """
    def score(self, predicted_tracks, true_tracks):
        """
        Calculate the precision, recall and f-score metrics for each user and find the average.
        Inputs must be dictionaries in the following format: {user_id_1: [track_id_1, track_id_2, ...], user_id_2: ...}
        :param predicted_tracks: dictionary of unique tracks predicted for each user
        :param true_tracks: dictionary of unique tracks listened by each user
        :return: average precision, recall and f-score metrics as floats
        """
        total_users = len(predicted_tracks)
        total_precision = 0
        total_recall = 0
        total_f_score = 0
        for user_id in predicted_tracks:
            user_predicted_tracks = predicted_tracks[user_id]
            user_true_tracks = [] if user_id not in true_tracks else true_tracks[user_id]
            TP = len(numpy.intersect1d(user_predicted_tracks, user_true_tracks))  # how many predicted items for user are relevant?
            TP_FP = len(user_predicted_tracks)  # how many items were predicted in total for user?
            TP_FN = len(user_true_tracks)  # how many relevant tracks were in total for user?
            user_precision = TP*1.0/TP_FP if not TP_FP == 0 else 1
            user_recall = TP*1.0/TP_FN if not TP_FN == 0 else 1
            total_precision += user_precision  # how many selected tracks are relevant for user?
            total_recall += user_recall  # how many relevant tracks are selected?
            if not (user_precision == 0 and user_recall == 0):
                total_f_score += 2*user_precision*user_recall/(user_precision+user_recall)
        return total_precision/total_users, total_recall/total_users, total_f_score/total_users

    def pprint_scores(self, (precision, recall, f_score), type):
        """
        Given precision, recall and f_score pprint them.
        :param type: Type of the model
        """
        print "==== %s ====" % type
        print "Model precision: %s" % precision
        print "Model recall: %s" % recall
        print "F1 score: %s" % f_score

