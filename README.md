## pytig

pytig is a python library to automatically generate images from text with a few commands using open source TIG's (text to image generation) algorithms. 

    import pytig as tig
    
    tig.download.datasets(name=['bird'])
    Bird = tig.generate("photosythnesis is a process", model='bird' )
    Bird.img.show()
   
   ![alt text](example_bird.PNG)

Traditionally TIG's are often developed by researchers, hoping to be published.  Once a paper is published, the algorithms are no longer maintained. Making it difficult for hobbyists, other researchers and engineers to replicate, experiment, and implement in production.

## Things you need to know before you install
1. Greater tham 4 GB of disk space for all the training data and various models.
2. python 3.6+
3. Time and patience.  It all depends on your machine but in general things run in minutes and hours not seconds.





TThe library facilates generating images fro text  using text to image generation algorithms (TIG's) developed by resea 






Programmatically generating images from text is a powerful concept with many practical applications in numerous fields such as education, engineering, game making. However in practice using these algorithms to generate images can be difficult i.e. [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf). The primary reasons are (1) Reseaserchers often write the code for the lab algorithms are written by researchers  

## Usage
`pip install pytig`

    import pytig as tig

    #test
    # No GPU required
    img = tig.generate("photosythnesis is a process", model='test' )

    #bird
    #coco




### What pytig does.
- Accept text as an input and return an image based on a tig maintained library with three inputs. 
    (1) predict text 
    (2) algorithm 
    (3) Process type i.e. CPU or GPU  
- Automate and Log model runs of tig maintained models.
- Help in preparing training data for new models.  A model is a trained object that when fed text can generate an image for that text.
        #### Example 1:
        A model trained on the cub data set (birds)  
- Share amd reproduce algorithms    

### What tig does not do.
- Supply the compute resources.


## Major Dependancies
pytig is dependant on large packages for preprocessing text, training and predicting images i.e. guild, spacy, textacy, pandas, pytorch

### Terminology
- algorithm - A set of rules and processes which output a model
- model - The algorithm's output which is used to generate the image

- inputs:
    - your train data or use premade example datasets
        -   Example:
        -   Example:
    - sentence(s) you wish to generate images for
        -   Example:
- output:
    -  Image(s)
        -   Example:





# Current Use Cases
- Generate random images of birds:
-
- Generate random images of Live objects
-

# Limitations
- Training Models on your own take a long time
- Lots of Resources
- Current pretrained are limited



# TIG Maintainad Algorithms
### AttnGAN

Pytorch implementation for reproducing AttnGAN results in the paper 

**Data**

1. Download our preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`



**Training**
- Pre-train DAMSM models:
  - For bird dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0`
  - For coco dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 1`

- Train AttnGAN models:
  - For bird dataset: `python main.py --cfg cfg/bird_attn2.yml --gpu 2`
  - For coco dataset: `python main.py --cfg cfg/coco_attn2.yml --gpu 3`

- `*.yml` files are example configuration files for training/evaluation our models.



**Pretrained Model**
- [DAMSM for bird](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V). Download and save it to `DAMSMencoders/`
- [DAMSM for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ). Download and save it to `DAMSMencoders/`
- [AttnGAN for bird](https://drive.google.com/open?id=1lqNG75suOuR_8gjoEPYNp8VyT_ufPPig). Download and save it to `models/`
- [AttnGAN for coco](https://drive.google.com/open?id=1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi). Download and save it to `models/`

- [AttnDCGAN for bird](https://drive.google.com/open?id=19TG0JUoXurxsmZLaJ82Yo6O0UJ6aDBpg). Download and save it to `models/`
  - This is an variant of AttnGAN which applies the propsoed attention mechanisms to DCGAN framework.

**Sampling**
- Run `python main.py --cfg cfg/eval_bird.yml --gpu 1` to generate examples from captions in files listed in "./data/birds/example_filenames.txt". Results are saved to `DAMSMencoders/`.
- Change the `eval_*.yml` files to generate images from other pre-trained models.
- Input your own sentence in "./data/birds/example_captions.txt" if you wannt to generate images from customized sentences.

**Validation**
- To generate images for all captions in the validation dataset, change B_VALIDATION to True in the eval_*.yml. and then run `python main.py --cfg cfg/eval_bird.yml --gpu 1`
- We compute inception score for models trained on birds using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).
- We compute inception score for models trained on coco using [improved-gan/inception_score](https://github.com/openai/improved-gan/tree/master/inception_score).


**Examples generated by AttnGAN [[Blog]](https://blogs.microsoft.com/ai/drawing-ai/)**

 bird example              |  coco example
:-------------------------:|:-------------------------:
![](https://github.com/taoxugit/AttnGAN/blob/master/example_bird.png)  |  ![](https://github.com/taoxugit/AttnGAN/blob/master/example_coco.png)


#### Creating an API
[Evaluation code](eval) embedded into a callable containerized API is included in the `eval\` folder.

### Citing AttnGAN
If you find AttnGAN useful in your research, please consider citing:

```
@article{Tao18attngan,
  author    = {Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He},
  title     = {AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks},
  Year = {2018},
  booktitle = {{CVPR}}
}
```

**Reference**

- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916) [[code]](https://github.com/hanzhanggit/StackGAN-v2)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) [[code]](https://github.com/carpedm20/DCGAN-tensorflow)


<a class="anchor" id="1"></a>
