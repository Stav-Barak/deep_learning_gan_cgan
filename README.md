# deep_learning_gan_cgan

This repository contains the implementation of Assignment 4 in the Deep Learning course at Ben-Gurion University.  
The project explores generative models on tabular data, focusing on Generative Adversarial Networks (GANs) and Conditional GANs (cGANs) using the Adult dataset.  

The assignment focuses on:
- Implementing GAN and cGAN architectures from scratch with PyTorch  
- Using autoencoders for embedding representations of tabular data  
- Training adversarial models with stability considerations (optimizers, learning rates, dropout, batch normalization)  
- Evaluating generative models with both detection and efficacy metrics  
- Comparing real vs. synthetic feature distributions through visualizations  

---

## Project Structure

- `gan_adult.ipynb` – Jupyter Notebook implementing the GAN model with autoencoder-based embeddings on the Adult dataset  
- `cgan_adult.ipynb` – Jupyter Notebook implementing the Conditional GAN with label-conditioned generation on the Adult dataset  
- `gan_cgan_report.pdf` – Report summarizing preprocessing, methodology, experiments, and results  

---

## Dataset

- Adult dataset (ARFF format): 32,561 samples, 15 features (6 numeric, 9 categorical)  
- Target feature: income (<=50K or >50K), with imbalanced distribution (about 76% vs. 24%)  
- Preprocessing:  
  - Continuous features scaled with MinMaxScaler  
  - Categorical features encoded with OneHotEncoder  
  - Combined into unified feature arrays for GAN input  

Train-test split: 80% / 20%, stratified on target label, repeated across 3 random seeds (1, 2, 3).  

---

## Model Architectures

### GAN
- Autoencoder: compresses real data into embeddings for GAN training  
- Generator: maps random noise to latent embeddings  
- Discriminator: differentiates between real vs. synthetic embeddings  
- Training losses:  
  - Autoencoder – MSE reconstruction loss  
  - Generator and Discriminator – BCEWithLogitsLoss adversarial loss  

### Conditional GAN (cGAN)
- Similar to GAN but conditioned on income label  
- Generator input: noise + label (one-hot)  
- Discriminator input: sample embedding + label (one-hot)  
- Conditional Autoencoder ensures reconstructions align with labels  
- Uses spectral normalization, dropout, and LeakyReLU for stability  

---

## Training Setup

- Optimizers: Adam (separate for generator, discriminator, autoencoder)  
  - gen_lr = 0.0001  
  - disc_lr = 0.00001  
- Noise dimensions: 100–200 tested  
- Batch sizes: 128, 256, 512 tested  
- Epochs: 30–50 with early stopping checks  
- Embedding dimensions: 32 (GAN), 64 (cGAN)  

---

## Results

### GAN (average across 3 seeds)
- Average Efficacy Ratio: 0.7729  
- Average Detection AUC: 1.0000 (synthetic distinguishable from real)  
- Synthetic data retained about 77 percent of predictive power of real dataset  
- Strengths: hours-per-week, relationship, race distributions  
- Weaknesses: capital-gain/loss spikes, smoother categorical distributions  

### cGAN (average across 3 seeds)
- Average Efficacy Ratio: 0.9157  
- Average Detection AUC: 1.0000  
- Synthetic data retained about 91.6 percent of predictive power of real dataset  
- Better alignment of categorical features (workclass, education, marital status, sex)  
- Some smoothing of rare distributions (capital gain/loss, education-num peaks)  

---

## Setup

Install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas seaborn
```

---

## Run

1. Open one of the notebooks:
- `gan_adult.ipynb`  
- `cgan_adult.ipynb`  

2. Run all cells to preprocess the data, train the model, and generate synthetic samples.  

3. Training and validation losses and generated samples will be displayed.  

4. The report (`gan_cgan_report.pdf`) includes comparisons of real vs. synthetic distributions, correlation heatmaps, and evaluation metrics.  
