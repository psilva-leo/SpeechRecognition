import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings("ignore")
words = ['go', 'back', 'right', 'left', 'stop']


class HMM(object):

    def __init__(self):

        def setup():

            def load_patterns(file):
                patterns = None
                sizes = np.zeros(len(words))
                counter = 0

                f = open(file, 'rb')
                data = f.readlines()

                stack = []
                for i in range(np.shape(data)[0]):
                    #data2 = list(data[i].split())
                    #data2 = map(float, data[i].split())
                    data2 = list(map(float, data[i].split()))
                    data2 = np.reshape(data2, (1, -1))
                    #data2 = list(np.reshape(data2, (1, -1)))
                    if i == 0:
                        stack = data2
                    else:
                        stack = np.vstack((stack, data2))

                f.close()
                sizes[counter] = np.shape(stack)[0]
                counter += 1

                if patterns is None:
                    patterns = stack
                else:
                    patterns = np.vstack((patterns, stack))

                return patterns

            hidden = 1

            self.go_model = GaussianHMM(n_components=hidden, covariance_type="diag", n_iter=10000).fit(
                load_patterns('go.bin'))

            self.back_model = GaussianHMM(n_components=hidden, covariance_type="diag", n_iter=10000).fit(
                load_patterns('back.bin'))

            self.right_model = GaussianHMM(n_components=hidden, covariance_type="diag", n_iter=10000).fit(
                load_patterns('right.bin'))

            self.left_model = GaussianHMM(n_components=hidden, covariance_type="diag", n_iter=10000).fit(
                load_patterns('left.bin'))

            self.stop_model = GaussianHMM(n_components=hidden, covariance_type="diag", n_iter=10000).fit(
                load_patterns('stop.bin'))

        setup()
        self.number_of_components = 5

    def match(self, pattern):

        probabilities = np.zeros(5)
        probabilities[0] = self.go_model.score(np.reshape(pattern, (1, -1)))
        probabilities[1] = self.back_model.score(np.reshape(pattern, (1, -1)))
        probabilities[2] = self.right_model.score(np.reshape(pattern, (1, -1)))
        probabilities[3] = self.left_model.score(np.reshape(pattern, (1, -1)))
        probabilities[4] = self.stop_model.score(np.reshape(pattern, (1, -1)))

        probabilities = abs(probabilities)

        index, error = min(enumerate(probabilities), key=lambda x: x[1])

        if error < 9500:
            if index == 0:
                return 0
            elif index == 1:
                return 1
            elif index == 2:
                return 2
            elif index == 3:
                return 3
            else:
                return 4
        return -1
