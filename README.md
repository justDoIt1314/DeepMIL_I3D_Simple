# DeepMIL Pytorch Version
Improved implemention of "Real-world Anomaly Detection in Surveillance Videos" CVPR2018

The feature extractor is pretrained I3D, which can be downloaded from [https://github.com/DavideA/c3d-pytorch](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc)

where we oversample each video frame with the “10-crop” augment, “10-crop” means cropping images into the center, four corners, and their mirrored counterparts. _0.npy is the center, _1~ _4.npy is the corners, and _5 ~ _9 is the mirrored counterparts. 


---

- **How to train**

  1. download or extract the features.
  2. use *make_list.py* in the *list* folder to generate the training and test list.
  3. change the parameters in option.py 
  4. run *main.py*

- **How to test**

  run *test.py* and the model is in the ckpt folder.

---


Thanks for your attention!
