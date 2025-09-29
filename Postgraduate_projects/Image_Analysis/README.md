## A7 Coursework: Image Analysis

This coursework explores 3 different topics, which are (1) classical image classification, (2) machine learning for reconstruction and (3) iamge quality metrics and exploring the pitfalls of using machine learning. 

Below are the main components of the repository :

- `report/`: Contains the final coursework report.
- `module_1.ipynb`: Contains code used in module 1 of the report
- `module_2.ipynb`: Contains code used in module 2 of the report
- `module_3.ipymb`: Contains code used in module 3 of the report
- `case_a.ipymb`: Notebooks given in the coursework guidelines, which are altered to answer questions in module 3 
- `case_b.ipymb`: Notebooks given in the coursework guidelines, which are altered to answer questions in module 3

These are some supplementary components:

- `results_for_plotting` : contains saved images and results that are used for plotting in the notebooks
- `data` : contains the data used in the jupyter notebooks
- `denoiser.pth` and `unet.py` : models given by the coursework guidelines
- `requirements.txt`: packages used to run the notebooks
- `license` : This project has a MIT license.

---

## Installation and running

Due to the large data files, building a Docker image can take 30+ minutes. To avoid this long build time, it is recommended to use a Python virtual environment.

1. Clone the repostiory 

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a7_coursework/jn492.git
```
2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Register the virtual environment as a jupyter kernel

```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```
5. Run notebook with kernel
---

### Note on running `case_a.ipynb` and `case_b.ipynb`

While all notebooks can be run locally, I **do not recommend** running `case_a.ipynb` and `case_b.ipynb` in this environment. These notebooks involve training machine learning models, which can be **very slow on CPU-only setups**.

To run them more efficiently:

1. Download the following files to your local machine or Google Drive:
  - `case_a.ipynb`
  - `case_b.ipynb`
  - `data/0_development_data.pkl`
  - `data/0_test_data.pkl`

2. Open the notebooks in **[Google Colab](https://colab.research.google.com/)** and **enable GPU support**

3. Adjust the file paths inside the notebooks to reflect where the `.pkl` files are located (e.g., `/content/` in Colab, or your mounted Google Drive directory).