function [corr] = xcorr_fft(queries, keys)
% XCORR_FFT Calculates the cross-correlation of the two input vectors
%   using the FFT-based method.


% Calculate cross-correlation
fftSignal1 = fft(queries, [], 2);
fftSignal2 = fft(keys, [], 2);
res = fftSignal1 .* conj(fftSignal2);
corr = ifft(res, [], 2);
end
