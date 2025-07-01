This model is designed for Subdivision Name Entity Recognition (NER) on OCR text results from historical deed documents. It is a customized NER solution built from scratch using the spaCy library with the 'en_core_web_md' English language pipeline.
The training dataset consists of sentences extracted via keyword matching from deed documents that have corresponding volunteer-annotated GeoJSON files. Specifically, the training labels are derived from the add_cov attribute in these GeoJSON files, which stores corrected subdivision names verified by volunteers.
The model was trained on the UMN MSI interactive-gpu cluster using the following configurations:
Base Model: spaCy's en_core_web_md
Training Epochs: 20
Optimizer: Adam with learning rate decay
Evaluation Metric: F1-score on subdivision names
The model achieved its optimal state after the 20th epoch.