![title.png](./plot/title.png)

<h1>Abstract</h1>
This paper presents a lightweight framework for generating structured chest X-ray reports by fine-tuning LLaVA-Med with LoRA on the IU X-Ray dataset. Our method achieves state-of-the-art performance across multiple metrics, demonstrating significant improvements in clinical accuracy and semantic coherence compared to existing approaches. To help readers quickly deploy our model, we write this instruction.

<h1>Get Started</h1>
<h2>1. Required Packages</h2>
Please install the required packages for our model via:

```Shell
git clone https://github.com/H1963977384/A-Fine-Tuned-LLaVA-Med-Framework-for-Automatic-Chest-X-ray-Report-Generation.git
cd A-Fine-Tuned-LLaVA-Med-Framework-for-Automatic-Chest-X-ray-Report-Generation
pip install -r requirements.txt
```

<h2>2. LLaVA-Med Deployment</h2>
<h3>(1) Clone this repository</h3>

```Python
git clone https://github.com/microsoft/LLaVA-Med.git
```

<h3>(2) Install relavant packages</h3>

```Shell
cd LLaVA-Med
pip install --upgrade pip
pip install --user -e .
```

<h2>3. Dataset</h2>
<h3>(1) Description</h3>
To achieve the research objectives, this study utilizes the IU X-Ray Dataset. Collected retrospectively between 2011 and 2018 by researchers at Indiana University Health from two large hospital systems within Indiana's patient care network, this dataset was specifically constructed for chest X-ray image understanding and report generation tasks.

![Dataset.png](./plot/dataset.png)

<h3>(2) Download and Transform</h3>
Load the image data by the following code. If encountering network problems, please download directly from [Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university).

```Shell
cd ../data
python download.py
```

The text data has been preprocessed and is ready for immediate use. The image paths and corresponding text data should be organized into separate JSON files for training and testing sets.

```Shell
python report2json.py
```

<h2>3. Training</h2>
The training hyperparameters (epochs, batch size, gradient accumulation steps, LoRA rank, and alpha) can be adjusted according to available computational resources.

```Shell
cd ..
python train_lora.py \
  --model_path microsoft/llava-med-v1.5-mistral-7b \
  --json_file ~/data/train_report.json \
  --image_dir ~/data/images \
  --output_dir ~/lora_weight \
  --epochs 10 \
  --batch_size 1 \
  --gradient_accumulation_steps 64 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lr 2e-4
```

<h2>4. Evaluation</h2>
<h3>(1) Only LLaVA-Med</h3>
Please ensure you have already downloaded the dataset and placed under **data** folder.

```Shell
python ./eval/llava.py \
  --json_file ~/data/test_report.json
```

![Only_LLaVA-Med.png](./plot/Only_LLaVA-Med.png)


<h3>(2) LLaVA-Med + LoRA</h3>
Users can either train their own LoRA weights or utilize our pre-trained versions. To use our weights, download them via:

```Shell
git clone https://github.com/H1963977384/LoRA_Weight.git
```

The LoRA weights can then be integrated into the base model through:

```Shell
python ./eval/llava_lora.py \
  --lora_path ~/LoRA_Weight
  --json_file ~/data/test_report.json
```

![LLaVA-Med+LoRA.png](./plot/LLaVA-Med+LoRA.png)

<h2>Main Result</h2>

![result.png](./plot/result.png)



<h1>Contribution</h1>
The team collaboratively completed this research. The seamless integration of each phase ensured the smooth progression of the research. <br>

![hyt.png](./plot/hyt.png)
![ln.png](./plot/ln.png)
![lry.png](./plot/lry.png)
![tsy.png](./plot/tsy.png)
![hjm.png](./plot/hjm.png)
