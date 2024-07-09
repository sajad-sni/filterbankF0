import Utilities as F
from FilterBankF0Tracking import f0tracking
import numpy as np
import pandas as pd
import matplotlib.backends.backend_pdf

def F0Table(path, Stimulus):
    subs = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
            'S16']

    # Read Stimulus
    stm = pd.read_csv('./stm/' + Stimulus + '.csv').values
    mse = []
    rmse = []
    acc5 = []
    # Read subject data
    f0s = np.zeros((len(subs), stm.shape[0]))
    amps = np.zeros((len(subs), stm.shape[0]))
    f0_amps = np.zeros((len(subs), stm.shape[0]))
    h1_amps = np.zeros((len(subs), stm.shape[0]))
    h2_amps = np.zeros((len(subs), stm.shape[0]))
    h3_amps = np.zeros((len(subs), stm.shape[0]))
    pdf = matplotlib.backends.backend_pdf.PdfPages('figures.pdf')
    for idx, sub in enumerate(subs):
        rec = pd.read_csv(path + Stimulus + '/' + sub + '.csv')
        recEven = rec.loc[rec['Polarity'] == 'Even'].iloc[:, 0:4096].values
        recOdd = rec.loc[rec['Polarity'] == 'Odd'].iloc[:, 0:4096].values
        rec = np.mean((recEven + recOdd) / 2, axis=0)
        # F0 Tracking
        f0, amp, evalParams, fig, har_amp = f0tracking(rec, stm, idx)
        pdf.savefig(fig)
        f0s[idx] = f0
        amps[idx] = amp
        f0_amps[idx] = har_amp[0]
        h1_amps[idx] = har_amp[1]
        h2_amps[idx] = har_amp[2]
        h3_amps[idx] = har_amp[3]
        mse.append(round(evalParams[0], 2))
        rmse.append(round(evalParams[1], 2))
        acc5.append(round(evalParams[2], 2))
        print('Subject{:3d}:  mse: {:6.2f}  rmse: {:6.2f}   Acc5: {:6.2f}  '.format(idx+1, evalParams[0],
                                                                                    evalParams[1], evalParams[2]))
    pdf.close()
    df_f0 = pd.DataFrame(f0s, index=subs)
    df_amp = pd.DataFrame(amps, index=subs)
    df_f0_amp = pd.DataFrame(f0_amps, index=subs)
    df_h1_amp = pd.DataFrame(h1_amps, index=subs)
    df_h2_amp = pd.DataFrame(h2_amps, index=subs)
    df_h3_amp = pd.DataFrame(h3_amps, index=subs)
    df_f0.to_excel('f0_'+Stimulus+'.xlsx', index=True, header=True)
    df_amp.to_excel('amp_'+Stimulus+'.xlsx', index=True, header=True)
    df_f0_amp.to_excel('f0_amp_' + Stimulus + '.xlsx', index=True, header=True)
    df_h1_amp.to_excel('h1_amp_' + Stimulus + '.xlsx', index=True, header=True)
    df_h2_amp.to_excel('h2_amp_' + Stimulus + '.xlsx', index=True, header=True)
    df_h3_amp.to_excel('h3_amp_' + Stimulus + '.xlsx', index=True, header=True)
    result = {
        'Subject': (np.arange(len(subs)) + 1),
        'MSE': mse,
        'RMSE': rmse,
        'acc5': acc5
    }

    # Save the file
    df = pd.DataFrame(data=result)
    df.loc[len(df.index)] = ['mean', round(np.mean(mse), 2), round(np.mean(rmse),2), round(np.mean(acc5), 2)]
    df.loc[len(df.index)] = ['var', round(np.var(mse), 2), round(np.var(rmse), 2), round(np.var(acc5), 2)]
    df.loc[len(df.index)] = ['std', round(np.std(mse), 2), round(np.std(rmse), 2), round(np.std(acc5), 2)]
    df.to_csv(Stimulus + '_result.csv', index=False)


