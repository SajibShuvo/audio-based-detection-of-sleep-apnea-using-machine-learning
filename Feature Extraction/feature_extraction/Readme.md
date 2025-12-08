# List of features with description
1. **Relative Difference Function (RDF)**

   a. According to Cho et al. [1], RDF is useful for onset detection of apnea events.\
   b. Basically, rdf = diff[log{RMS(audio signal)}], where diff is the differentiation operation.\
   c. Extracted features variable names and meanings:
    - rdf_mean = Mean value of RDF
    - rdf_std = Standard deviation of RDF
    - rdf_skew = Skewness of RDF
    - rdf_kurt = Kurtosis of RDF
    - rdf_max = Maximum value of RDF
    - rdf_min = Minimum value of RDF
    - rdf_pos_ratio = Fraction of the RDF values that are positive (denotes energy increment)
    - rdf_deriv_std = Standard deviation of the derivative of RDF
    - rdf_deriv_mean = Mean of the derivative of RDF

3. The ratio of the spectral energy above 800 Hz to that below 800 Hz [2]. It is denoted by **PR800**.

4. **Mel Frequency Cepstrum Coefficients (MFCC):**

    According to [1], MFCC is “A short-term power spectrum based on the nonlinear mel scale of frequency. This is concisely describes the overall shape of a spectral envelop, and is commonly used as feature in speech recognition and music information retrieval such as genre classification.” The extracted MFCC features can be divided into two broad categories - general MFCC and constant Q-based MFCC.


    **a. General MFCC:** 13 MFCC coefficients were calculated using librosa.feature.mfcc() function. Audio was resampled to 16 kHz and a 25 ms window with 10 ms hop was used. \The variables shown below are the overall average value of the derivative of the Nth MFCC coefficient in the format of MFCC_Overall_AverageN.
    - MFCC_Overall_Average2
    - MFCC_Overall_Average3
    - MFCC_Overall_Average4
    - MFCC_Overall_Average9
    - MFCC_Overall_Average10
    - MFCC_Overall_Average12
      
    **b. Constant Q-based MFCC:** The overall average of area-method (2nd central) moments of constant-Q based MFCCs are calculated. 


5. **Linear Predictive Coding:** According to [1], it is the “Spectral envelope based on the information  of a linear predictive model.”
    - LPC_env_mean = Mean over all frames and freqs of spectral envelope
    - LPC_env_std = Standard deviation over all frames and freqs of spectral envelope
    - LPC_freq_deriv_mean = Mean of frequency-derivative (envelope differential along frequency)
    - LPC_freq_deriv_std = Standard deviation (std) of frequency-derivative
    - LPC_delta_0_mean = Overall mean of temporal delta of envelope for width 0
    - LPC_delta_0_std = Overall std of temporal delta of envelope for width 0
    - LPC_delta_3_mean = Overall mean of temporal delta of envelope for width 3
    - LPC_delta_3_std = Overall std of temporal delta of envelope for width 3
    - LPC_delta_4_mean = Overall mean of temporal delta of envelope for width 4
    - LPC_delta_4_std = Overall std of temporal delta of envelope for width 4
    - LPC_delta_5_mean = Overall mean of temporal delta of envelope for width 5
    - LPC_delta_5_std = Overall std of temporal delta of envelope for width 5

**6. Fraction of Low Energy Windows:** According to [1], “This is a good measure of how much of a signal is quiet relative to the rest of a signal.” The short time energy window length was 20 ms and the hop duration was 10 ms. The duration of larger analysis segment was 1 second.  
    - overall_fraction = Overall value of the fraction over the entire audio clip.
    - fraction_std = Standard deviation of the fraction over the entire audio clip.
    - derivative_mean = Mean of the derivative of the fraction over the entire audio clip.
    - derivative_std  = Standard deviation  of the derivative of the fraction over the entire audio clip.  

**7. Mel spectrogram:** It is calculated using librosa.feature.melspectrogram() function. 512 points FFT is calculated with a window length of 400 samples and hop length of 160 samples at 16 kHz (resampled from 48 kHz). The derivative and the double derivative of the mel spectrogram is also calculated. For each audio segment, 3 channels of spectrogram data is calculated (main mel spectrogram, derivative, double derivative).



## References:

 [1]	S.-W. Cho, S. J. Jung, J. H. Shin, T.-B. Won, C.-S. Rhee, and J.-W. Kim, “Evaluating Prediction Models of Sleep Apnea From Smartphone-Recorded Sleep Breathing Sounds,” JAMA Otolaryngol. Neck Surg., vol. 148, no. 6, p. 515, June 2022, doi: 10.1001/jamaoto.2022.0244. \
[2]	S. Cao, I. Rosenzweig, F. Bilotta, H. Jiang, and M. Xia, “Automatic detection of obstructive sleep apnea based on speech or snoring sounds: a narrative review,” J. Thorac. Dis., vol. 16, no. 4, pp. 2654–2667, Apr. 2024, doi: 10.21037/jtd-24-310. 