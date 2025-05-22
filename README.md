### BlindMRI : Stress Modeling with Generative Models & LLMs

Welcome to the BlindMRI repository — a modular and extensible pipeline for stress detection and simulation, built with l GANs, LLM validation, and domain-informed data augmentation. This project is geared toward advancing stress state modeling, particularly in visually impaired and MRI-restricted environments.

## 📁 Repository Structure and Usage
This repository is organized into three major directories, each reflecting a key phase in the stress modeling lifecycle:

   - blindMRI_baseline/ :
              Initial modeling based on physiological signals and demographic metadata.

   - blindMRI_env/ :
              Stress detection enhanced with environmental and MRI contextual factors.

   - blindMRI_fine_tuned/ :
              Final dataset and models fine-tuned for blind patient-specific scenarios.

Each folder includes:

Full pipeline: data collection, preprocessing, LLM validation, cGAN-based data generation, and evaluation.

Intermediate + final datasets with tracking of changes.

Plot exports, metrics logs, and CSVs for reproducibility and traceability.

  - A .env template for securely storing your OpenAI API key.

  - A requirements.txt listing all Python dependencies.

  - A README.md summarizing statistical logic.

  - A PDF detailing every stage and model used in the pipeline.

The full code is to be integrated into an AI-agent to have full control on the pipline but  to explain the process clearly the hole pipeline is guided and checked manually to have track of every output before heading to next stage.
In the last stage, the handeling the llm generation and checking of data besides the training GAN's files are supposed to be independant for each feature but the process is the same for stage 2 and 3 so they are gathered in one file each and the hole process in detail is detailed in the PDFs.
## 🛠️ How to Set Up Locally
1.Clone the repository

    git clone https://github.com/waad64/BlindMRI.git
    cd BlindMRI
    
2.Create and activate a virtual environment
  On Windows:
  
    python -m venv venv
    .\venv\Scripts\activate
    
  On macOS/Linux:

    python3 -m venv venv
    source venv/bin/activate
    
3.Install dependencies

    pip install -r requirements.txt
    
4.Set up your OpenAI API key
 -Create a .env file in the root directory (if not already there).

   Add your key like this:
   
          OPENAI_API_KEY=your-api-key-here
 
   Don't have one? Generate it form : [https://openai.com](https://openai.com/api/)


🔍 For in-depth documentation, refer to the *.pdf files inside each directory — they walk through the full data/LLM/GAN lifecycle.
Due t tha large volume of the dataset, each version for each stage is saved in a ZIP file, to see you you need to download it .  
