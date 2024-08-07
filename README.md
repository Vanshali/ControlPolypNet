# ControlPolypNet: Towards Controlled Colon Polyp Synthesis for Improved Polyp Segmentation

Paper Link: [https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Sharma_ControlPolypNet_Towards_Controlled_Colon_Polyp_Synthesis_for_Improved_Polyp_Segmentation_CVPRW_2024_paper.html](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Sharma_ControlPolypNet_Towards_Controlled_Colon_Polyp_Synthesis_for_Improved_Polyp_Segmentation_CVPRW_2024_paper.html)

## 1. Introduction

In recent years, generative models have been very popular in medical imaging applications because they generate realistic-looking synthetic images, which is crucial for the medical domain. These generated images often complement the hard-to-obtain annotated authentic medical data because acquiring such data requires expensive manual effort by clinical experts and raises privacy concerns. Moreover, with recent diffusion models, the generated data can be controlled using a conditioning mechanism, simultaneously ensuring diversity within synthetic samples. This control can allow experts to generate data based on different scenarios, which would otherwise be hard to obtain. However, how well these models perform for colonoscopy still needs to be explored. Do they preserve clinically significant information in generated frames? Do they help in downstream tasks such as polyp segmentation? Therefore, in this work, we propose ControlPolypNet, a novel stable diffusion based framework. We control the generation process (polyp size, shape and location) using a novel custom-masked input control, which generates images preserving important endoluminal information. Additionally, our model comprises a detection module, which discards some of the generated images that do not possess lesion characterising features, ensuring clinically relevant data. We further utilize the generated polyp frames to improve performance in the downstream task of polyp segmentation. Using these generated images, we found an average improvement of 6.84% and 1.3% (Jaccard index) on the CVC-ClinicDB and Kvasir-SEG datasets, respectively.

### Objective at a Glance!
![Polyp Generation](figures/intro.png)
*Figure 1:  Controlling polyp generation using custom masks while leveraging largely accessible non-polyp/negative images. We turned negative samples into positive ones with controlled polyp shape, size and location.*

### General Comparison with Conventional Approaches
![Comparison](figures/intro2_controlnet.svg)
*Figure 2: Augmentation strategies; (a) Conventional augmentation techniques present limited diversity among samples, (b) Conventional generative approaches use all generated images irrespective of their clinical relevance, and (c) Our approach has an additional detection step that selects generated images which are detected with a high confidence score, ensuring clinical relevance.*

## 2. ControlPolypNet
- *ControlPolypNet* consists of three main parts: (a) Stable Diffusion U-Net architecture loaded with pre-trained weights of SD v1-5, (b) ControlNet, and (c) YOLOv8, a detector pre-trained on the polyp images. 
- To make the model learn the mapping $N' \rightarrow P'$, we prepared our training set such that initially, it learns $M \rightarrow P$.
![ControlPolypNet](figures/controlnet_diag1.svg)
*Figure 3: The proposed framework uses custom-masked images as input control with a ``polyp" text prompt. The pre-processing pipeline shows the elimination of uninformative negative frames. Custom masks are used to generate polyps during the evaluation phase of \textit{ControlPolypNet}. The generated polyp images are fed to a YOLOv8 detector that selects clinically significant frames with a confidence score $>=0.7$.*
The repository will be updated soon!
