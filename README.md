# CESN (Conceptor Echo State Networks)

This project contains implementations of various neural network models, including:

- CESN (Conceptor Echo State Networks)
- ESN (Echo State Networks)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- BP (Back Propagation Neural Network)

## Overview

This repository implements several types of neural networks with a focus on Conceptor Echo State Networks (CESN). The implementation includes both traditional neural networks (BP) and various types of recurrent neural networks (ESN, LSTM, GRU). The code is particularly useful for time series prediction and pattern recognition tasks.

## File Description

### Main Models
- `CESN.m`: Main implementation of Conceptor Echo State Networks
- `C2_ESN_concept_delma.m`: Example implementation of ESN concepts
- `LSTMwind.m`: LSTM model implementation
- `D1_6_GRU_del.m`: GRU model implementation
- `D1_5_LSTM_del.m`: LSTM model implementation
- `BP.m`: Back Propagation Neural Network implementation

### Utility Functions
- `function/scurve.m`: S-curve dataset generation
- `function/lle.m`: Locally Linear Embedding implementation
- `function/swissroll.m`: Swiss roll dataset generation
- `function/xcorr_fft.m`: Fast cross-correlation using FFT
- `function/seriesDecomp.m`: Time series decomposition
- `function/nrmse.m`: Normalized Root Mean Square Error calculation

## Requirements

- MATLAB R2019b or higher
- Neural Network Toolbox
- Deep Learning Toolbox (optional)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/zhuyingqin/Transfer-Concept-ESN.git
```

2. Add the project directory to your MATLAB path:
```matlab
addpath(genpath('Transfer-Concept-ESN'));
```

## Usage Instructions

1. Ensure your MATLAB environment meets the above requirements
2. Download all files to the same directory
3. Run the corresponding .m files in MATLAB

### Example Usage

```matlab
% Example code for using CESN
load('traindata.mat');  % Load your training data
model = CESN(input_size, reservoir_size, output_size);
output = model.train(input_data, target_data);

% Example of using utility functions
% Generate S-curve dataset
[X, labels] = scurve(1000, 0.1);

% Calculate NRMSE
error = nrmse(predicted, actual);

% Perform time series decomposition
[trend, seasonal, residual] = seriesDecomp(timeseries_data);
```

## Notes

- Some code may require training data, please ensure data files are in the correct path
- Execution time may be lengthy, please be patient
- It is recommended to read the comments in the code to understand parameter settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{CESN2024,
  author = {Yingqin Zhu},
  title = {Transfer-Concept-ESN},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/zhuyingqin/Transfer-Concept-ESN}
}
```

## Contact

- Author: Yingqin Zhu
- Project Link: [https://github.com/zhuyingqin/Transfer-Concept-ESN](https://github.com/zhuyingqin/Transfer-Concept-ESN) 