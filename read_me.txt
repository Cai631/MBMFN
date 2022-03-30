Our algorithm is based on IMDN（Lightweight Image Super-Resolution with Information Multi-distillation Network)

#IMDN：
[[arXiv]](https://arxiv.org/pdf/1909.11856v1.pdf)
[[Poster]](https://github.com/Zheng222/IMDN/blob/master/images/acmmm19_poster.pdf)
[[ACM DL]](https://dl.acm.org/citation.cfm?id=3351084)

# [ICCV 2019 Workshop AIM report](https://arxiv.org/abs/1911.01249)
The simplified version of IMDN won the **first place** at Contrained Super-Resolution Challenge (Track1 & Track2). The test code is available at [Google Drive](https://drive.google.com/open?id=1BQkpqp2oZUH_J_amJv33ehGjx6gvCd0L)


Special statement:
  Because our algorithm is based on the IDMN algorithm, we didn't change the name in the code, for example, python test_IMDN.py, we didn't change the file name of test_IMDN.py. 
We just use their names, the rest of the content is completely different from IDMN.

## Testing
Pytorch 1.1
* Runing testing:
```bash
# Set5 x2 IMDN
python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/Set5/x2 --checkpoint checkpoints/epoch_x2.pth --upscale_factor 2


```
* Calculating  FLOPs and parameters, input size is 192*192
```bash
python calc_FLOPs.py
```

## Training
* Download [Training dataset DIV2K](https://drive.google.com/open?id=12hOYsMa8t1ErKj6PZA352icsx9mz1TwB)
* Convert png file to npy file
```bash
python scripts/png2npy.py --pathFrom /path/to/DIV2K/ --pathTo /path/to/DIV2K_decoded/
```
* Run training x2, x3, x4 model
Put '' DIV2K_decoded'' in the current folder
```bash
python train_IMDN.py --root ./ --scale 2 --pretrained checkpoints/IMDN_x2.pth

