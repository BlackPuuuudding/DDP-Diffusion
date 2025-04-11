# Decoupled Dual-Path Diffusion: Precise Spatial-Semantic Modeling for Human-Object Interaction Generation

This is the repository for the paper "Decoupled Dual-Path Diffusion: Precise Spatial-Semantic Modeling for Human-Object Interaction Generation".

## Results

<table>
  <thead>
    <tr>
      <th rowspan="2"> </th>
      <th colspan="4">FGAHOI Swin-Tiny↑</th>
      <th colspan="4">FGAHOI Swin-Large↑</th>
    </tr>
    <tr>
      <th>Full</th><th>Rare</th><th>Full</th><th>Rare</th>
      <th>Full</th><th>Rare</th><th>Full</th><th>Rare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>DDP-Diffusion</strong></td>
      <td><strong>30.73</strong></td><td><strong>24.06</strong></td><td><strong>32.29</strong></td><td><strong>25.84</strong></td>
      <td><strong>32.25</strong></td><td><strong>26.18</strong></td><td><strong>33.47</strong></td><td><strong>27.52</strong></td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th rowspan="2"> </th>
      <th colspan="4">RLIPv2-Tiny↑</th>
      <th colspan="4">RLIPv2-Large↑</th>
      <th rowspan="2">FID↓</th>
      <th rowspan="2">KID↓</th>
    </tr>
    <tr>
      <th>Full</th><th></th><th>Rare</th><th></th>
      <th>Full</th><th></th><th>Rare</th><th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>DDP-Diffusion</strong></td>
      <td><strong>35.37</strong></td><td></td><td><strong>28.87</strong></td><td></td>
      <td><strong>37.60</strong></td><td></td><td><strong>32.36</strong></td><td></td>
      <td><strong>17.35</strong></td><td><strong>0.00496</strong></td>
    </tr>
  </tbody>
</table>

  <table>
  <thead>
    <tr>
      <th rowspan="2"> </th>
      <th colspan="6">Image Text Consistency↑</th>
    </tr>
    <tr>
      <th>Similarity</th><th>Sim@0.75</th><th>Sim@0.8</th><th>Sim@0.85</th><th>Sim@0.9</th><th>Sim@0.95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>DDP-Diffusion</strong></td>
      <td><strong>0.7194</strong></td>
      <td><strong>41.32%</strong></td>
      <td><strong>20.92%</strong></td>
      <td><strong>6.89%</strong></td>
      <td><strong>2.17%</strong></td>
      <td><strong>0.25%</strong></td>
    </tr>
  </tbody>
</table>

DDP-Diffusion evaluates **Interaction Controllability** using the HOI Detection Score, and measures **Interaction Accuracy** through the proposed Action Score. More details about the evaluation protocols can be found in the paper.

## Download  DDP-Diffusion model



## Reproduce & Evaluate

1. In `inference_batch.py`, replace `ckpt.pth` with the selected checkpoint.

2. Perform inference using DDP-Diffusion to generate the HICO-DET test set based on ground truth.

      ```bash
      python inference_batch.py --batch_size 1 --folder generated_output --seed 489 --scheduled-sampling 1.0 --half
      ```

3. Setup FGAHOI. See [FGAHOI repo](https://github.com/xiaomabufei/FGAHOI) on how to setup FGAHOI.

4. Setup RLIPv2. See [RLIPv2 repo](https://github.com/JacobYuan7/RLIPv2?tab=readme-ov-file) on how to setup RLIPv2.

5. Prepare for evaluate on FGAHOI. See `id_prepare_inference.ipynb`

6. Evaluate on FGAHOI.

      ```bash
      python main.py --backbone swin_tiny --dataset_file hico --resume weights/FGAHOI_Tiny.pth --num_verb_classes 117 --num_obj_classes 80 --output_dir logs  --merge --hierarchical_merge --task_merge --eval --hoi_path data/id_generated_output --pretrain_model_path "" --output_dir logs/id-generated-output-t
      ```

7. Evaluate for FID and KID. We recommend to resize hico_det dataset to 512x512 before perform image quality evaluation, for a fair comparison. We use [torch-fidelity](https://github.com/toshas/torch-fidelity).

      ```bash
      fidelity --gpu 0 --fid --isc --kid --input2 ~/data/hico_det_test_resize  --input1 ~/FGAHOI/data/data/id_generated_output/images/test2015
      ```

## Training

1. Run the following command:

      ```bash
      torchrun --nproc_per_node=2 main.py --yaml_file configs/hoi_hico_text.yaml --ckpt <existing_gligen_checkpoint> --name test --batch_size=4 --gradient_accumulation_step 2 --total_iters 500000 --amp true --disable_inference_in_training true --official_ckpt_name <existing SD v1.4/v1.5 checkpoint>
      ```

## Acknowledgement

This work is developed based on the codebase of [GLIGEN](https://github.com/gligen/GLIGEN) , [LDM](https://github.com/CompVis/latent-diffusion) and [InteractDiffusion](https://github.com/jiuntian/interactdiffusion).
