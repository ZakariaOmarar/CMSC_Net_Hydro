# Feature Reference For Mode Detection

This document explains the feature groups extracted from each DataSegment window, what each feature means, and why it can help mode detection (Pump, Standstill, Turbine).

## 1. Data Contract Context

A DataSegment window contains:

- Mic channels: high-rate acoustic signals.
- Accel channels: low-rate vibration amplitude signals.
- Optional metadata from ingestion (for example dominant vibration frequency traces).

Feature extractors transform each window into scalar or vector descriptors. Vector features are flattened downstream.

## 2. Time-Domain Features (TimeDomainExtractor)

Applied per channel (mic and accel).

Feature keys per channel name X:

- X_rms
- X_peak
- X_crest_factor
- X_kurtosis
- X_skewness
- X_zero_crossing_rate

Definitions:

- RMS: signal energy level, $\sqrt{\frac{1}{N}\sum x^2}$.
- Peak: max absolute amplitude.
- Crest factor: peak / RMS, sensitive to impulsive behavior.
- Kurtosis: tail heaviness (impulsiveness).
- Skewness: asymmetry around mean.
- Zero crossing rate: sign-change frequency proxy.

Why useful for mode detection:

- Pump and Turbine can differ in energy distribution and impulsiveness.
- Standstill often shows lower dynamic activity.

## 3. Frequency-Domain Features (FrequencyDomainExtractor)

Applied per channel (mic and accel) via FFT magnitude/power.

Feature keys per channel name X:

- X_spectral_centroid
- X_spectral_rolloff
- X_spectral_bandwidth
- X_spectral_flatness
- X_fundamental_amplitude
- X_vpf_amplitude
- X_harmonic_ratio_2x
- X_harmonic_ratio_3x
- X_harmonic_ratio_4x
- X_band_energy_0_50
- X_band_energy_50_200
- X_band_energy_200_1k
- X_band_energy_1k_5k

Definitions:

- Spectral centroid: energy-weighted mean frequency.
- Spectral rolloff: frequency below which 85 percent of power lies.
- Spectral bandwidth: spread around centroid.
- Spectral flatness: noise-like vs tonal content.
- Fundamental amplitude: amplitude near configured f0 (default 5.867 Hz).
- VPF amplitude: vane-pass amplitude near configured 117.3 Hz.
- Harmonic ratios: harmonic amplitude divided by f0 amplitude.
- Band energies: power integrated in predefined bands.

Why useful:

- Machine modes differ strongly in tonal lines and harmonic structure.

## 4. Time-Frequency Features (TimeFrequencyExtractor)

Applied per channel (default mic only, can include accel).

Feature keys per channel name X:

- X_stft_mean_power (vector)
- X_mel_spectrogram (vector)
- X_mfcc (vector)
- X_spectral_contrast (vector)

Definitions:

- STFT mean power: average frequency-bin energy over time.
- Mel spectrogram mean: perceptual band energy summary.
- MFCC mean: compact spectral envelope representation.
- Spectral contrast mean: difference between spectral peaks and valleys across bands.

Why useful:

- Captures non-stationary and timbral differences between operation modes.

## 5. Vibration Envelope Features (VibrationEnvelopeExtractor)

Applied per accel channel. Output is one concatenated vector per channel.

Feature key per accel channel name X:

- X_vibration_features (vector of 18 values)

Vector layout:

- Time part (11):
  - rms, mean_abs, peak, peak_to_peak, crest_factor, kurtosis, skewness, std,
    mean_abs_rate_of_change, max_abs_rate_of_change, linear_slope
- Spectral part (3):
  - total_energy, spectral_centroid, sub_sync_energy_ratio
- Transition part (4):
  - max_abs_gradient, monotonic_up_flag, monotonic_down_flag, rolling_variance_delta

Why useful:

- Vibration envelopes are highly informative for low-frequency mechanical state.

## 6. Vibration Dominant-Frequency Features (VibrationFrequencyExtractor)

Applied per accel channel using metadata field vibration_frequencies (provided by ingestion adapter and window-aligned by window index and overlap).

Feature key per accel channel name X:

- X_vib_freq_stats (vector of 8 values)

Vector layout:

- mean
- std
- min
- max
- p10
- p50
- p90
- slope

Why useful:

- Mode differences can appear as shifts or trends in dominant vibration frequency.
- This can improve separability especially when amplitude alone is ambiguous.

## 7. Cross-Channel Features (CrossChannelExtractor)

Cross-sensor relational descriptors.

Feature keys:

- mic_correlation_matrix (upper triangle vector)
- accel_correlation_matrix (upper triangle vector)
- generator_vs_turbine_energy
- mic_coherence_mean
- mic_coherence_per_pair (vector)
- acoustic_dominance
- structural_indicator
- flow_fault_indicator
- subsurface_indicator
- channel_energy_distribution (vector)

Definitions:

- Correlation vectors: pairwise linear dependence between channels.
- Generator vs turbine energy ratio: compares grouped mic energy.
- Coherence features: frequency-domain synchronization between mic pairs at key frequencies.
- Acoustic dominance and derived indicators: heuristic cross-modal balance metrics.
- Channel energy distribution: normalized per-channel energy profile.

Why useful:

- Mode often changes spatial and cross-modal coupling patterns, not only single-channel amplitudes.

## 8. TDoA Features (TDoAExtractor)

Based on GCC-PHAT pairwise delays between microphones.

Feature keys:

- gcc_phat_matrix (upper triangle vector)
- tdoa_pair_i_j for each mic pair
- tdoa_pair_i_j_confidence for each mic pair

Definitions:

- TDoA values estimate relative arrival time shifts.
- Confidence reflects quality of the delay estimate.

Why useful:

- Different running modes can shift dominant acoustic source geometry and delay patterns.

## 9. Practical Notes For Mode Accuracy

- Some groups are high-dimensional (for example time_frequency), so normalization and split strategy matter.
- Perfect scores with window-level random splits can be optimistic due to leakage across windows from the same recording.
- Prefer group-aware splits by recording identity whenever possible.
- Use feature-combination reports to identify minimal strong subsets (better robustness and easier deployment).

## 10. Feature Groups Used In Search

The mode feature search script evaluates combinations of these groups:

- time_domain
- frequency_domain
- time_frequency
- vibration_envelope
- vibration_frequency
- cross_channel
- tdoa

Output report fields:

- best combination
- top ranked combinations
- leave-one-group-out impact
- add-one-group gain
- split diagnostics and warnings
