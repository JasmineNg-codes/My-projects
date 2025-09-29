## A2 Coursework

This coursework explores 3 different topics learnt in the Medical Imaging modules, which are PET/CT reconstructions, MRI denoising, and CT segmentation respectively. Below is the structure of the repository:

- `report/`: Contains the final coursework report.
- `module_1/workbook_1`: Contains the workbook for module_1
- `module_2/workbook_1`: Contains the workbook for module_1
- `module_3/workbook_1`: Contains the workbook for module_1

---

## Installation

1. Git clone the repository and cd into the folder

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a2_coursework/jn492.git

cd jn492
```

2. Build docker image (this may take some time, as there are large files in module_3)

```bash
docker build -t a2_coursework .
```

3. Run the docker image
```bash
docker run -p 8888:8888 a2_coursework
```

3. Click on the link in your terminal

- click on a link that looks similar to: http://127.0.0.1:8888/tree. Now you're ready to run all the notebooks!

## Contributions
Please feel free to fork this repository and submit a pull request

## License
This project is licensed under the MIT License, see license.txt.





