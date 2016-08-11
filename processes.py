import nengo
import numpy as np
from scipy import fftpack
import hmm


class FFTProcessTrain(nengo.Process):
    def __init__(self, sound, sample_rate, frame_size):
        super(FFTProcessTrain, self).__init__()
        self.sound = sound
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.current = 0

    def make_step(self, shape_in, shape_out, dt, rng):
        def step(t):
            frame_step = self.frame_size/4
            signal = self.sound[self.current:self.current+self.frame_size]
            if len(signal) < self.frame_size:
                size = self.frame_size - len(signal)
                zeros = np.zeros(size)
                signal = np.append(signal, zeros)

            coeff = 0.97
            atenuation = [x * coeff for x in signal[:-1]]
            atenuation = np.subtract(signal[1:], atenuation)
            signal = np.append(signal[0], atenuation)

            signal = signal*np.hamming(self.frame_size)
            signal = fftpack.fft(signal)
            signal = abs(signal)/(self.frame_size/2.)
            signal = signal[:int(self.frame_size/2)]

            signal /= 300.      # signal in range [0,1]

            self.current += frame_step
            return signal

        return step


class FFTProcess(nengo.Process):
    def __init__(self, sample_rate, frame_size):
        super(FFTProcess, self).__init__()
        self.sound = np.zeros(10)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.current = 0
        self.currentLine = 0
        self.readLine = True

    def load_audio(self):
        audio = None

        f = open('out.bin', 'rb')
        data = f.readlines()

        if (np.shape(data)[0] - 1) == self.currentLine and self.readLine is True:
            try:
                audio = map(int, data[-1].split())
                print(np.shape(audio))
            except ValueError:
                print('Error loading audio. Try again.')

            self.readLine = False
            self.current = 0

        f.close()
        return audio

    def make_step(self, shape_in, shape_out, dt, rng):
        def step(t):
            if t == 0.001:
                print('Nengo started.')

            frame_step = self.frame_size/4

            audio = self.load_audio()

            if audio is not None:
                self.sound = audio
                print('Recognizing...')
                print(len(self.sound))

            signal = self.sound[self.current:self.current+self.frame_size]

            if len(signal) < self.frame_size:
                size = self.frame_size - len(signal)
                zeros = np.zeros(size)
                signal = np.append(signal, zeros)

                if self.readLine is False:
                    print('Recognition finished')
                    self.readLine = True
                    self.currentLine += 1

            coeff = 0.97
            atenuation = [x * coeff for x in signal[:-1]]
            atenuation = np.subtract(signal[1:], atenuation)
            signal = np.append(signal[0], atenuation)

            signal = signal*np.hamming(self.frame_size)
            signal = fftpack.fft(signal)
            signal = abs(signal)/(self.frame_size/2.)
            signal = signal[:int(self.frame_size/2)]

            signal /= 300.      # signal in range [0,1]

            self.current += frame_step
            return signal

        return step


class ASRInput(object):
    def __init__(self):
        self.count = 0
        self.buffer = np.zeros(50 * 36)     # MFCC(12) + Delta1(12) + Delta2(12)
        self.c_buffer = np.zeros((5, 12))       # MFCC
        self.d_buffer = np.zeros((5, 12))       # Delta1
        self.melbuffer = np.zeros((3, 160))
        self.HMM = hmm.HMM()
        self.filterbank = self.mel()
        self.prediction_buffer = []
        self.space_buffer = np.zeros(20) - 1
        self.command = 5    # Stop

    def step(self, t, x):

        # Get current MFCC
        self.melbuffer[-1] = x
        self.melbuffer = np.roll(self.melbuffer, -5, axis=0)
        current_mfcc = self.mel_coeffs(np.sum(self.melbuffer, axis=0))


        # MFCC
        self.c_buffer[-1] = current_mfcc
        self.c_buffer = np.roll(self.c_buffer, 1, axis=0)

        # Delta1
        if 1 < self.count < 4:  # from 2 to 3
            self.d_buffer[-1] = self.c_buffer[1]  # two positions before
        else:   # start at 4
            self.d_buffer[-1] = self.c_buffer[0] - self.c_buffer[4]  # two positions before and two after
        self.d_buffer = np.roll(self.d_buffer, 1, axis=0)

        # Delta2
        if self.count < 6:
            current_delta2 = np.zeros(12)
        else:   # start at 6
            current_delta2 = self.d_buffer[0] - self.d_buffer[4]

        # Compound data
        if self.count >= 6:
            result = np.append(self.c_buffer[4], self.d_buffer[2])
            result = np.append(result, current_delta2)
        else:
            result = np.append(current_mfcc, current_mfcc)
            result = np.append(result, current_mfcc)
        self.buffer = np.roll(self.buffer, -36)
        self.buffer[-36:] = result


        # test speech recognition
        vocab = self.HMM.match(self.buffer)
        self.space_buffer[-1] = vocab
        self.space_buffer = np.roll(self.space_buffer, 1)

        if vocab != -1:
            self.prediction_buffer = np.append(self.prediction_buffer, vocab).astype(int)

        if all(x == -1 for x in self.space_buffer) and (len(self.prediction_buffer) != 0):
            counts = np.bincount(self.prediction_buffer)
            self.command = vocab = np.argmax(counts)
            if vocab == 0:
                vocab = 'GO'
            elif vocab == 1:
                vocab = 'BACK'
            elif vocab == 2:
                vocab = 'RIGHT'
            elif vocab == 3:
                vocab = 'LEFT'
            else:
                vocab = 'STOP'

            print(str(self.count)+ ' : ' + str(vocab))
            self.prediction_buffer = []

        self.count += 1

        return self.command

    def load_patterns(self):
        f = open('pattern.bin', 'rb')
        data = f.readlines()
        data2 = []
        for i in range(np.shape(data)[0]):
            data2 = map(float, data[i].split())
            print(np.shape(data2))
        f.close()

        return data2

    def mel(self):
        def melfunc(frequency):
            return 1125 * np.log(1 + (frequency / 700))

        def imelfunc(frequency):
            return 700 * (np.exp(frequency / 1125) - 1)

        lower = 100.
        mlower = melfunc(lower)
        upper = 4000.
        mupper = melfunc(upper)
        number_of_filters = 26
        size = 320  # size of the FFT total
        samplerate = 8000

        mfreqs = np.linspace(mlower, mupper, number_of_filters + 2)
        freqs = imelfunc(mfreqs)
        index = np.floor((size + 1) * freqs / samplerate)

        filterbank = []
        for m in range(number_of_filters + 1):
            filter = np.zeros(size / 2)
            for k in range(size / 2):
                if k < index[m - 1]:
                    filter[k] = 0
                elif index[m - 1] <= k <= index[m]:
                    filter[k] = (k - index[m - 1]) / (index[m] - index[m - 1])
                elif index[m] <= k <= index[m + 1]:
                    filter[k] = (index[m + 1] - k) / (index[m + 1] - index[m])
                else:
                    filter[k] = 0
            if m == 0:
                filterbank = filter
            else:
                filterbank = np.vstack((filterbank, filter))

        return filterbank

    def mel_coeffs(self, signal):
        size = 320
        signal = np.where(signal == 0, np.finfo(float).eps, signal)     # transforms zeros values to the minimum float value possible
        coefficients = np.zeros(np.shape(self.filterbank)[0])
        import math
        for i in range(np.shape(self.filterbank)[0]):
            filtered = signal * self.filterbank[i]
            coefficients[i] = abs(sum(filtered) if not math.isnan(sum(filtered)) else 0.)
            coefficients[i] = np.where(coefficients[i] == 0, np.finfo(float).eps, coefficients[i])

        # print('coeff: ', coefficients)
        logcoeff = 10.*np.log10(coefficients)

        dct = fftpack.dct(logcoeff)
        # print('dct coeff: ', dct[:12])

        return dct[:12]


class ASRInputTrain(object):
    def __init__(self, file):
        self.count = 0
        self.sample = 0
        self.buffer = np.zeros(50 * 36)     # MFCC(12) + Delta1(12) + Delta2(12)
        self.c_buffer = np.zeros((5, 12))       # MFCC
        self.d_buffer = np.zeros((5, 12))       # Delta1
        self.melbuffer = np.zeros((3, 160))
        self.filterbank = self.mel()
        self.prediction_buffer = [-1, -1, -1, -1, -1]
        self.command = 5    # Stop
        self.file = file

    def step(self, t, x):

        # Get current MFCC
        self.melbuffer[-1] = x
        self.melbuffer = np.roll(self.melbuffer, -5, axis=0)
        current_mfcc = self.mel_coeffs(np.sum(self.melbuffer, axis=0))


        # MFCC
        self.c_buffer[-1] = current_mfcc
        self.c_buffer = np.roll(self.c_buffer, 1, axis=0)

        # Delta1
        if 1 < self.count < 4:  # from 2 to 3
            self.d_buffer[-1] = self.c_buffer[1]  # two positions before
        else:   # start at 4
            self.d_buffer[-1] = self.c_buffer[0] - self.c_buffer[4]  # two positions before and two after
        self.d_buffer = np.roll(self.d_buffer, 1, axis=0)

        # Delta2
        if self.count < 6:
            current_delta2 = np.zeros(12)
        else:   # start at 6
            current_delta2 = self.d_buffer[0] - self.d_buffer[4]

        # Compound data
        if self.count >= 6:
            result = np.append(self.c_buffer[4], self.d_buffer[2])
            result = np.append(result, current_delta2)
        else:
            result = np.append(current_mfcc, current_mfcc)
            result = np.append(result, current_mfcc)
        self.buffer = np.roll(self.buffer, -36)
        self.buffer[-36:] = result


        # Write in file
        if self.count == (166 * (self.sample + 1) + 38 * self.sample):
            self.sample += 1
            f = open(self.file, 'ab')
            print('buffer:', len(self.buffer))
            for i in range(len(self.buffer)):
                f.write('%f ' % self.buffer[i])
            f.write('\n')
            f.close()

        self.count += 1

        return self.command

    def load_patterns(self):
        f = open('pattern.bin', 'rb')
        data = f.readlines()
        data2 = []
        for i in range(np.shape(data)[0]):
            data2 = map(float, data[i].split())
            print(np.shape(data2))
        f.close()

        return data2

    def mel(self):
        def melfunc(frequency):
            return 1125 * np.log(1 + (frequency / 700))

        def imelfunc(frequency):
            return 700 * (np.exp(frequency / 1125) - 1)

        lower = 100.
        mlower = melfunc(lower)
        upper = 4000.
        mupper = melfunc(upper)
        number_of_filters = 26
        size = 320  # size of the FFT total
        samplerate = 8000

        mfreqs = np.linspace(mlower, mupper, number_of_filters + 2)
        freqs = imelfunc(mfreqs)
        index = np.floor((size + 1) * freqs / samplerate)

        filterbank = []
        for m in range(number_of_filters + 1):
            filter = np.zeros(size / 2)
            for k in range(size / 2):
                if k < index[m - 1]:
                    filter[k] = 0
                elif index[m - 1] <= k <= index[m]:
                    filter[k] = (k - index[m - 1]) / (index[m] - index[m - 1])
                elif index[m] <= k <= index[m + 1]:
                    filter[k] = (index[m + 1] - k) / (index[m + 1] - index[m])
                else:
                    filter[k] = 0
            if m == 0:
                filterbank = filter
            else:
                filterbank = np.vstack((filterbank, filter))

        return filterbank

    def mel_coeffs(self, signal):
        size = 320
        signal = np.where(signal == 0, np.finfo(float).eps, signal)     # transforms zeros values to the minimum float value possible
        coefficients = np.zeros(np.shape(self.filterbank)[0])
        import math
        for i in range(np.shape(self.filterbank)[0]):
            filtered = signal * self.filterbank[i]
            coefficients[i] = abs(sum(filtered) if not math.isnan(sum(filtered)) else 0.)
            coefficients[i] = np.where(coefficients[i] == 0, np.finfo(float).eps, coefficients[i])

        # print('coeff: ', coefficients)
        logcoeff = 10.*np.log10(coefficients)

        dct = fftpack.dct(logcoeff)
        # print('dct coeff: ', dct[:12])

        return dct[:12]