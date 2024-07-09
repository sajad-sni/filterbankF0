import pandas as pd
from scipy.io import wavfile
from FilterBankF0Tracking import f0tracking
import matplotlib
import matplotlib.backends.backend_pdf


pdf = matplotlib.backends.backend_pdf.PdfPages('figures.pdf')
Stimulus = 'FH'
stm = pd.read_csv('./stm/' + Stimulus + '.csv').values

samplerate, stm_wav = wavfile.read('D:/PHD/Stimulus Files/44.1kHz scaled/balloonS15Ha.wav')

f0, amp, evalParams, fig, har_amp = f0tracking(stm_wav, stm, 1)

pdf.savefig(fig)
pdf.close()

df_f0 = pd.DataFrame(f0[None, :], index=[Stimulus])
df_amp = pd.DataFrame(amp[None,:], index=[Stimulus])
df_f0_amp = pd.DataFrame(har_amp[0][None,:], index=[Stimulus])
df_h1_amp = pd.DataFrame(har_amp[1][None,:], index=[Stimulus])
df_h2_amp = pd.DataFrame(har_amp[2][None,:], index=[Stimulus])
df_h3_amp = pd.DataFrame(har_amp[3][None,:], index=[Stimulus])
df_f0.to_excel('f0_'+Stimulus+'.xlsx', index=True, header=True)
df_amp.to_excel('amp_'+Stimulus+'.xlsx', index=True, header=True)
df_f0_amp.to_excel('f0_amp_' + Stimulus + '.xlsx', index=True, header=True)
df_h1_amp.to_excel('h1_amp_' + Stimulus + '.xlsx', index=True, header=True)
df_h2_amp.to_excel('h2_amp_' + Stimulus + '.xlsx', index=True, header=True)
df_h3_amp.to_excel('h3_amp_' + Stimulus + '.xlsx', index=True, header=True)