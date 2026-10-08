# Beyond Softmax Confidence: Do Alternative Reliability Measures Justify Their Cost in Wearable Human Activity Recognition?
<img width="1376" height="677" alt="Softmax_arch" src="https://github.com/user-attachments/assets/a4cc7e49-dfbf-465e-9ea6-2cdb9f11471e" />
This repository implements the methodology proposed in the paper “When Is Softmax Confidence Sufficient for Trustworthy Wearable Human Activity Recognition?”

## Paper Overview
**Abstract:** Deep learning-based human activity recognition (HAR) typically relies on softmax outputs for activity classification, with the resulting probabilities also providing inexpensive measures of prediction confidence. Beyond these conventional measures, various reliability estimation approaches have been investigated using temporal evidence, prediction consistency, and test-time augmentation. However, whether such additional information provides meaningful improvements over simple softmax-based confidence measures remains unclear, particularly when accounting for computational cost. <br>
<br>
This study systematically compares output-based, temporal, augmentation-based, and composite reliability measures across four wearable HAR datasets, three backbone architectures, and 16 controlled sensor-corruption conditions. Their effectiveness is evaluated through error detection, selective prediction, and computational overhead. <br>
<br>
Under sensor corruption, predictive entropy achieves the best mean reliability performance across the three primary metrics on UCI-HAR and PAMAP2, while softmax-derived measures also perform best on WISDM. On MotionSense, the full composite improves mean error-detection AUROC from 0.7564 to 0.7648 and achieves better selective prediction at 50--70% coverage. However, predictive entropy retains better mean AUPRC and AURC. Furthermore, the full composite increases inference latency by approximately 308--345% in the UCI-HAR Raspberry Pi 4 evaluation. <br>
<br>
These findings demonstrate that incorporating additional reliability information does not consistently improve the identification of incorrect HAR predictions. The benefits of more complex reliability estimation depend on the dataset and evaluation criterion, highlighting the importance of assessing whether improvements justify their additional computational cost. <br>

## Dataset
This repository does not include datasets. Please download them from the official sources below and configure the dataset path accordingly.
- **UCI-HAR** dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **PAMAP2** dataset is available at https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- **MotionSense** dataset is available at https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- **WISDM** dataset is available at https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset

## Codebase Overview
The `model/` directory contains the three UCI-HAR backbone implementations. Each file includes one model architecture and the training and reliability-evaluation code used in the experiments.

- `model_cnn.py`: CNN backbone
- `model_bilstm.py`: BiLSTM backbone
- `model_transformer.py`: Transformer backbone

Dataset loading and preprocessing are not included.

## Citing this Repository
If you use this code in your research, please cite:
```bibtex
@article{Softmax,
  title   = {Beyond Softmax Confidence: Do Alternative Reliability Measures Justify Their Cost in Wearable Human Activity Recognition?},
  author  = {Seokyeong Jeon and Myung-Kyu Yi},
  journal = {},
  volume  = {},
  number  = {},
  pages   = {},
  year    = {2026},
  publisher = {}
}
```

## Contact
For questions or issues, please contact:
- Seo-Kyeong Jeon : seo1270@gachon.ac.kr
