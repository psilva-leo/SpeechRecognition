import sys
from PyQt4 import QtGui
import pyaudio
import pyqtgraph as pg
from scipy import arange
import scipy.fftpack
import numpy as np
import wave
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.fftpack import fft, rfft, ifft
from scipy import fromstring, roll

FORMAT = pyaudio.paInt16
CHANNELS = 1
FS = 8000
CHUNK = 1024

terminated = False
fftSpec = []


class myMainWindow(QtGui.QMainWindow):
    def closeEvent(self, event):
        global terminated
        terminated = True
        event.accept()


def mainLoop():
    global terminated, fftSpec

    # start Recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=FS, input=True,
                        frames_per_buffer=CHUNK)

    ### Application Creation

    ### Main window
    mainWindow = myMainWindow()
    mainWindow.setWindowTitle("Spectrum Analyzer")  # Title
    mainWindow.resize(1300, 500)  # Size
    ### Campus
    centralWid = QtGui.QWidget()
    mainWindow.setCentralWidget(centralWid)
    layH = QtGui.QHBoxLayout()
    centralWid.setLayout(layH)

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
    origWave = waveWid.getPlotItem()
    origWave.setMouseEnabled(y=False)  # to not be moved to the y-axis direction
    origWave.setYRange(-10000, 10000)
    origWave.setXRange(0, 512, padding=0)
    ### Axis
    specAxis = origWave.getAxis("bottom")
    specAxis.setLabel("Samples")
    LeftlayV.addWidget(waveWid)

    ### Spectrum display widget
    fftWid = pg.PlotWidget(title="FFT")
    fftItem = fftWid.getPlotItem()
    fftItem.setMouseEnabled(y=False)  # to not be moved to the y-axis direction
    fftItem.setYRange(0, 3000)
    fftItem.setXRange(0, FS/2, padding=0)
    ### Axis
    specAxis = fftItem.getAxis("bottom")
    specAxis.setLabel("Frequency [Hz]")
    LeftlayV.addWidget(fftWid)

    ### Spectogram
    specWid = pg.PlotWidget()
    specItem = pg.ImageItem()
    specWid.addItem(specItem)

    img_array = np.zeros((100, CHUNK / 2))

    # bipolar colormap
    pos = np.array([0., 1., 0.5, 0.25, 0.75])
    color = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [0, 0, 0, 255], (0, 0, 255, 255), (255, 0, 0, 255)],
                     dtype=np.ubyte)
    cmap = pg.ColorMap(pos, color)
    lut = cmap.getLookupTable(0.0, 1.0, 256)

    # set colormap
    specItem.setLookupTable(lut)
    specItem.setLevels([-50, 40])

    # setup the correct scaling for y-axis
    freq = np.arange((CHUNK / 2) + 1) / (float(CHUNK) / FS)
    yscale = 1.0 / (img_array.shape[1] / freq[-1])
    specItem.scale((1. / FS) * CHUNK, yscale)

    specWid.setLabel('left', 'Frequency', units='Hz')
    RightlayV.addWidget(specWid)

    ### Window display
    mainWindow.show()
    frames = []

    open('recorded.bin', 'w').close()
    file = open('recorded.bin', 'a')
    while not terminated:
        orig = np.array([])

        # get audio samples
        data = stream.read(CHUNK)
        frames.append(data)

        orig = fromstring(data, dtype="int16")

        # Transform to frequency domain (FFT)
        originalfft = np.array(fft(orig))
        fftSpec = abs(originalfft) / (CHUNK / 2)
        fftSpec = fftSpec[:int(CHUNK / 2)]

        xf = 1.0 * np.arange(0, FS / 2., FS / (1.*CHUNK))

        # Spectrogram
        img_array = np.roll(img_array, -1, 0)
        img_array[-1:] = 10.0 * np.log10(fftSpec)

        # Plotting Graphs
        origWave.plot(orig, clear=True)
        fftItem.plot(xf, fftSpec, clear=True)
        specItem.setImage(img_array, autoLevels=False)
        QtGui.QApplication.processEvents()

    stream.stop_stream()
    stream.close()
    audio.terminate()
    file.close()

    waveFile = wave.open('recorded.wav', 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(FS)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


if __name__ == '__main__':
    mainLoop()
