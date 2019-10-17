import matplotlib.pyplot as plt
import numpy as np


def plot_1p(path, z, f, winlen, ref, title, y_label):
    plt.ioff()
    plt.figure()
    z_fft = 20*np.abs(np.log10(np.fft.fftshift(np.fft.fft(z[0:winlen] / np.max(z[0:winlen])))/ref))
    plt.semilogx(f, z_fft, lw=1, label='z')
    plt.xlabel('frequency (Hz)', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend(fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlim(10 ** 2, 10 ** 4)
    plt.ylim(20, 180)
    plt.tight_layout()
    plt.savefig(path + '.pdf')

    return


def plot_2p(path, z, y, f, winlen, ref, title, y_label):
    plt.ioff()
    plt.figure()

    z_fft = 20 * np.abs(np.log10(np.fft.fftshift(np.fft.fft(z[0:winlen] / np.max(z[0:winlen]))) / ref))
    y_fft = 20 * np.abs(np.log10(np.fft.fftshift(np.fft.fft(y[0:winlen] / np.max(y[0:winlen]))) / ref))

    plt.subplot(2, 1, 1)
    plt.title(title, fontsize=14)
    plt.xlim(10 ** 2, 10 ** 4)
    #plt.ylim(20, 180)
    plt.semilogx(f, z_fft, lw=1, label='z')
    plt.ylabel(y_label, fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(2, 1, 2)
    plt.xlim(10 ** 2, 10 ** 4)
    #plt.ylim(20, 180)
    plt.semilogx(f, y_fft, lw=1, label='y')
    plt.xlabel('frequency (Hz)', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(path + '.pdf')

    return
