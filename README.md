# EEG-based decoding of the spatial focus of auditory attention using CSPs

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations. By downloading and/or installing this software and associated files on your computing system you agree to use the software under the terms and condition as specified in the License agreement.

If this code has been useful for you, please cite [1] and/or [2].

## About

This repository includes the MATLAB-code for the subject-specific CSP decoder as explained in [1] and subject-specific RGC algorithm as explained in [2]. mainSubjectSpecific.m contains the main script for [CSP/mainSubjectSpecific.m](CSP) and [RGC/mainSubjectSpecific.m](RGC), which works by default with the dataset published in https://zenodo.org/record/3997352#.X0y1B3kzZEY. The [preprocessData.m](preprocessData.m)-script can be used to preprocess the data, downloaded from the aforementioned link, and replaces the function of the same name at https://zenodo.org/record/3997352#.X0y1B3kzZEY. 

Developed and tested in MATLAB R2020a.

Note: Tensorlab is required (https://www.tensorlab.net/).

### Quick guide

1. Download the dataset from https://zenodo.org/record/3997352#.X0y3nHkzZEZ.
2. Run [preprocessData.m](preprocessData.m).
3. Run [CSP/mainSubjectSpecific.m](mainSubjectSpecific.m).
4. Add your own datasets and play around!

## Contact
Simon Geirnaert  
KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data Analytics  
KU Leuven, Department of Neurosciences, Research Group ExpORL  
Leuven.AI - KU Leuven institute for AI  
<simon.geirnaert@esat.kuleuven.be>

Tom Francart
KU Leuven, Department of Neurosciences, Research Group ExpORL  
<tom.francart@kuleuven.be>

Alexander Bertrand
KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data Analytics  
Leuven.AI - KU Leuven institute for AI  
<alexander.bertrand@esat.kuleuven.be>

 ## References
 
[1] S. Geirnaert, T. Francart and A. Bertrand, "Fast EEG-Based Decoding Of The Directional Focus Of Auditory Attention Using Common Spatial Patterns," in IEEE Transactions on Biomedical Engineering, vol. 68, no. 5, pp. 1557-1568, May 2021, doi: 10.1109/TBME.2020.3033446.

[2] S. Geirnaert, T. Francart and A. Bertrand, "Riemannian Geometry-Based Decoding of the Directional Focus of Auditory Attention Using EEG," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 1115-1119, doi: 10.1109/ICASSP39728.2021.9413404.