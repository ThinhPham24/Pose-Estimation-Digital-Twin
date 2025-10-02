# This algorithm includes:
- Use the zero-shot learning Segmentation to identify the object. 
- Use the zero-shot learning Depth estimation to estimate the depth of the object.
- Use the ICP algorithm (different methods) to estimate the pose of the object.

# Installations
```
pip install -r requirements.txt
```

# There is workflow.

<center>
<img src='Assets/Algorithm_v2.gif'>
</center>

# Results:
<center>
<img src='Assets/demo.gif'>
</center>

<center>
<img src='Pose_results/result.gif'>
</center>

# Some code have been borrowed by Refs

1. [Deep Learning-Based Object Pose Estimation: A Comprehensive Survey](https://github.com/CNJianLiu/Awesome-Object-Pose-Estimation)

2. [YOLOE: Real-Time Seeing Anything](https://arxiv.org/abs/2503.07465).

3. [Depth Anything V2](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.pdf)

4. [DepthFM: Fast Monocular Depth Estimation with Flow Matching](https://github.com/CompVis/depth-fm/tree/main).

5. [HRC](https://github.com/WanqingXia/HRC_DetAnyPose)

6. [https://arxiv.org/pdf/2409.02920](https://arxiv.org/pdf/2409.02920).

7. [RoboTwin](https://github.com/TianxingChen/RoboTwin?tab=readme-ov-file).

8. [Sim2Real](https://arxiv.org/pdf/2409.10784).
9. [Large vision-language models enabled novel objects 6D pose estimation for human-robot collaboration](https://download.ssrn.com/rcim/4ed6743f-372b-4878-8dec-c397fbf38289-meca.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFgU4sR7xTfKf0nLANznsOQvtAzPCjdppRVQIZohDFPPAiAFGrTapm8cNCqfeQ5Nw8aD8HIlXsPAN7RInuPnWURipCq%2BBQg%2BEAQaDDMwODQ3NTMwMTI1NyIM%2Fgrh%2BJvu%2FB80hblIKpsFmPaxh10YJam12%2FF4oJfoc9WySMrRoaDhabimnxuIBpkwBGOrdGeb6yEWvgBdMaYOP5scq9ejUmQbWSZ%2BPDpuXxtZBYKk5Z2H8VzyWQHDocK2wgfxdP7HGTIX%2Fwdt44wISvlF5sMp1xLseXOaVD59uUBKey1UZHy0nS%2BmNdoYKGsVX7sN9%2F1uQd6BADRSBfspvJrsH9E1AKPN5AWzdgt7TD8X%2FUXg49LOWGw7khLGrdLE%2F4H8UqDIzE28m3%2BRMbcVsMa%2FTTZrVfArSHE%2FXmoyM5d%2BvQNgFQOMDbuTgPBY6%2FveGE%2BGzahp%2FGBuoITy2aG6r%2FFKQe8EBlhkLCLT4XEvAfQ1q6s%2BliSiq0wWAE6%2BkFmKPxam5YLkAMLvvnjIOd542oy%2FPWQlMUecpCAtPZdr8sfQ4XT%2FTcCUpKMYhXrbkQPN7jC9P83eWoS42IWJfcLE6iPNsfDOqORcaEBjZPj0gsfos%2BJ0TQCcH6zhN3i9s%2FqDMUi0AA0VOt5cVqT3T7Dr7dYatGVfHe3NRlKbVWw7T68XoedvUWS%2FBAOhP31r4fjxv7tWZ2Shq5yxB1CGNrEBYmGIAzGs7FNnMyBympkDt0g1A8kst2muOuR8kaprHl03IxR6L%2FJRyPeDLUA0SR70OX5Jf%2FwF26ylwERNXid%2BmCuNfhaambUQQ17lfvw%2BBbWvmfTvjh3%2FgXa2ntVHz1bE95Mj3F%2BXZMr3MIGMweRNJRoO9UNRD0dSeYEla6lpi9R1v0C0IginWj11sQ4B%2F%2FuUlWY27QfBqFnpXr0aUhsbmCrVT3nRi4YzjyBOvieC1E37Szkq2q2HZBHfTCmP30aOy9gwGn0Tiu1T7bwcWOBIj%2FWUo21axTGwPu6jkRH%2BJ2I3aZD1pM2AcLXWpzCkyd6%2BBjqyATxLx74qgISJej%2BlEdL4REjFCkw6CJgM87S1n5D56bN7gGS4rWuLUWh6U2Nbv4BY0e1D3QzPQtWEAnjLGNrKnjjkDL%2BVQHo90GMh4xrzfmMcdx7ujIzvRwkgbEBpbaStfHaTW%2BxAF8aZvV1UEiEy4Zf9BSCilViZhxhQBTWqxBY8xGe17cBaD7TJos1hxJ3%2F%2B69v6xRtwFZKk4ph71KYYdlIUSEoKncM%2FWBK9uHktbmd464%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250317T045541Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWEQ7IONRX2%2F20250317%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=40da81106ade831a4b082eadbfebb79271a4e6966bb95ed1890dae15e4c6e497&abstractId=5042919) 

10. [DEFOMStereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo)

11. [MonSter](https://github.com/Junda24/MonSter/tree/main)

12. [Stereo Deep learning lists](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

12. [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)

13. [FoundationPose](https://github.com/NVlabs/FoundationPose)

14. [Keymatching](https://github.com/fhagelskjaer/keymatchnet)

15. [ZePHyr](https://github.com/r-pad/zephyr)

# Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{quan2025cadpose,
  title   = {CAD-guided 6D pose estimation with deep learning in digital twin for industrial collaborative robot manipulation},
  author  = {Dong, Quang Huan and Pham, The-Thinh and Nguyen, Tuan-Khanh and Tran, Chi-Cuong and Tran, Hoang-Huy and Do-Duy, Tan and Nguyen, Khang Hoang Vinh and Nguyen, Quang-Chien},
  journal = {EAI Endorsed Transactions on AI and Robotics},
  year    = {2025},
  doi     = {10.4108/airo.9676}
}





