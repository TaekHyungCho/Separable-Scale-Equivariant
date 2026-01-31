# Separable Scale-Equivariant Object Perception
Source code for article entitled ["Scale-Equivariant Object Perception for Autonomous Driving"](https://ieeexplore.ieee.org/document/10480255).  
In this repository, we briefly introduce our code and provide additional information to enhance clarity.

## Environment
### Prerequisites
1. mmcv-full == 2.0.1
2. mmdetection >= 3.1.0
    * This repository only includes modified parts of the mmdetection. For complete details, please clone the corresponding version.
3. mmengine
4. KITTI Dataset

### Data
Please run the following code to generate data pickle files.
```bash
python tools/data_converter/kitti_converter.py --data_path ${RAW_DATA_PATH} --save_path ${PICKLE_SAVE_PATH}
```
### Scale-Equivariant Basis Functions 

To utilize scale-equivariant convolutions, you must configure the basis functions as follows:

* **Scale-Equivariant Steerable Basis Functions**  
    * Pre-calculated basis functions can be generated using the functions in plugin/utils/basis.py.

    * To use these, specify the storage path and filename in the save_dir and filename of the backbone parameters within the config file.

    * **Note**: If you set these parameters to None, the system is designed to automatically generate and implement the basis 
    functions during the initialization of the training process.

* **Discrete Scale-Equivariant Basis Functions**

    * Unlike the steerable version, this method requires pre-calculated basis functions. Please refer to the [DISCO](https://github.com/ISosnovik/disco) repository to prepare them.

    * You must specify the valid save_dir and filename in the configuration file to load these prepared functions.

### Training
  * ### Object Detection  

    Please run tools/run_train.sh, for example:
    ```bash
    bash tools/run_train.sh
    ```
    In the configuration files, you can modify the permutation options, which control the type of basis functions used:

    1. Scale ***combined*** basis functions: Fuse scales in each scale dimension.
    2. Scale ***isolated*** basis functions: Use one scale in each scale dimension.

    You can set it up for your purpose.

  * ### Object Tracking (Re-Identification)

    Please run the tracking_reid.train module with specific configuration files. For example:
    ```bash
    python -m tracking_reid.train --cfg_path ${CONFIG_FILES}
    ```

    These modules require pre-trained backbone networks trained on the object detection task described above. You can specify the saved path of the pre-trained backbone network in the configuration file.

    You can also choose basis permutation options to select the basis format, such as scale-combined or scale-isolated.

### Erratum and Supplementary Material
After our paper was accepted in IEEE Transactions on Intelligent Vehicles (IEEE T-IV), we identified subtle implementation deviations in the backbone configuration that were not fully addressed in the original article.  

For transparency, we have included experimental results from both the original scale-isolated and our scale-combined approaches in the attached **Supplementary Material**.   
These findings are available in the enclosed file for your reference. 

These comprehensive results offer a balanced view of performance differences, enhancing the clarity and integrity of our research. 

The source code in this repository has been updated to rectify key deviations.

### NEWS  
* 2024 / Mar : Our paper entitled "Scale-Equivariant Object Perception for Autonomous Driving" accepted in IEEE Transactions on Intelligent Vehicles (IEEE T-IV) 

* 2025 / Sep : Our paper published in IEEE T-IV, Volume: 10, Issue: 9, pp. 4361-4370

* 2026 / Feb : Released Code and Supplementary Material detailing implementation-specific deviations and their impact on scale-equivariance error and downstream task performance. 

### Acknowledgements
We express our appreciation to the authors of [SESN](https://github.com/ISosnovik/sesn) and [DISCO](https://github.com/ISosnovik/disco) for their excellent research.   
We also thanks to the [MMDetection](https://github.com/open-mmlab/mmdetection) team for facilitating our implementation.