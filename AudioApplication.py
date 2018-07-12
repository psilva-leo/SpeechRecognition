import sys
import _thread as thread
from PyQt4 import QtGui
from PyQt4.QtCore import QThread
import pyaudio
import pyqtgraph as pg
import neuralModel
import numpy as np
import wave
from scipy.fftpack import fft
from scipy import fromstring


FORMAT = pyaudio.paInt16
CHANNELS = 1
FS = 8000
CHUNK = 1024

terminated = False
recording = False
switch = 0

def nengo_thread_func():
    global recording, switch
    last_switch = 0
    # while not terminated:
    #     if not recording and (switch != last_switch):
    #         time.sleep(0.1)     # If there is no delay, Nengo tries to open the file while the other is writing
    last_switch = switch
    nengo_thread = NengoThread()
    nengo_thread.start()


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUI()
        self.mainLoop()

    def setupUI(self):

        self.setWindowTitle("Audio Application")  # Title
        self.resize(1300, 500)  # Size

        ### Widgets
        centralWid = QtGui.QWidget()
        self.setCentralWidget(centralWid)
        layV = QtGui.QVBoxLayout()
        centralWid.setLayout(layV)

        btn_record = QtGui.QPushButton('Record')

        def clickedfunc():
            global recording, switch
            if recording:
                btn_record.setStyleSheet("background-color: None")
                btn_record.setText('Record')
                recording = False
                switch += 1

            else:
                btn_record.setStyleSheet("background-color: red")
                btn_record.setText('Recording')
                recording = True

        btn_record.clicked.connect(clickedfunc)

        topWidget = QtGui.QWidget()
        layV.addWidget(topWidget)
        layV.addWidget(btn_record)

        layH = QtGui.QHBoxLayout()
        topWidget.setLayout(layH)

        LeftWidget = QtGui.QWidget()
        RightWidget = QtGui.QWidget()
        layH.addWidget(LeftWidget)
        layH.addWidget(RightWidget)

        LeftlayV = QtGui.QVBoxLayout()
        LeftWidget.setLayout(LeftlayV)

        RightlayV = QtGui.QVBoxLayout()
        RightWidget.setLayout(RightlayV)

        ### Original Wave display widget
        waveWid = pg.PlotWidget(title="Original Wave")
        self.origWave = waveWid.getPlotItem()
        self.origWave.setMouseEnabled(y=False)  # to not be moved to the y-axis direction
        self.origWave.setYRange(-10000, 10000)
        self.origWave.setXRange(0, 512, padding=0)
        ### Axis
        specAxis = self.origWave.getAxis("bottom")
        specAxis.setLabel("Samples")
        LeftlayV.addWidget(waveWid)

        ### Spectrum display widget
        fftWid = pg.PlotWidget(title="FFT")
        self.fftItem = fftWid.getPlotItem()
        self.fftItem.setMouseEnabled(y=False)  # to not be moved to the y-axis direction
        self.fftItem.setYRange(0, 3000)
        self.fftItem.setXRange(0, FS / 2, padding=0)
        ### Axis
        specAxis = self.fftItem.getAxis("bottom")
        specAxis.setLabel("Frequency [Hz]")
        LeftlayV.addWidget(fftWid)

        ### Spectogram
        specWid = pg.PlotWidget()
        self.specItem = pg.ImageItem()
        specWid.addItem(self.specItem)

        self.img_array = np.zeros((100, CHUNK // 2))

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [0, 0, 0, 255], (0, 0, 255, 255), (255, 0, 0, 255)],
                         dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.specItem.setLookupTable(lut)
        self.specItem.setLevels([-50, 40])

        # setup the correct scaling for y-axis
        freq = np.arange((CHUNK / 2) + 1) / (float(CHUNK) / FS)
        yscale = 1.0 / (self.img_array.shape[1] / freq[-1])
        self.specItem.scale((1. / FS) * CHUNK, yscale)

        specWid.setLabel('left', 'Frequency', units='Hz')
        RightlayV.addWidget(specWid)
        self.show()

    def mainLoop(self):
        global terminated, switch

        # start Recording
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=FS, input=True,
                            frames_per_buffer=CHUNK)

        # Cleaning the file. In case of a force quit, the file keep the last run data
        open('out.bin', 'wb').close()

        last_switch = 0
        sound = []
        while not terminated:
            if recording:
                # get audio samples
                data = stream.read(CHUNK)

                orig = fromstring(data, dtype="int16")
                sound = np.append(sound, orig)
                # Transform to frequency domain (FFT)
                # reduce noise and transform back to time
                originalfft = np.array(fft(orig))
                fftSpec = abs(originalfft) / (CHUNK / 2)
                fftSpec = fftSpec[:int(CHUNK / 2)]

                xf = 1.0 * np.arange(0, FS / 2., FS / (1. * CHUNK))

                # Spectrogram
                self.img_array = np.roll(self.img_array, -1, 0)
                self.img_array[-1:] = 10.0 * np.log10(fftSpec)

                # Plotting Graphs
                self.origWave.plot(orig, clear=True)
                self.fftItem.plot(xf, fftSpec, clear=True)
                self.specItem.setImage(self.img_array, autoLevels=False)
            else:
                if last_switch != switch:
                    last_switch = switch
                    frames = []

                    # Write in file
                    #f = open('out.bin', 'ab')
                    f = open('out.bin', 'a')

                    for i in range(len(sound)):
                        f.write('%d ' % sound[i])
                    f.write('\n')
                    f.close()

                    sound = []

            QtGui.QApplication.processEvents()

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def closeEvent(self, event):
        global terminated
        terminated = True
        event.accept()


class NengoThread(QThread):
    def __init__(self):
        QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        neuralModel.run()


if __name__ == '__main__':
    thread.start_new_thread(nengo_thread_func, ())
    app = QtGui.QApplication(sys.argv)
    win = MainWindow()


