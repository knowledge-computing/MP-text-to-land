import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import json
import time
import torch
import os

# check availability of GPU
def get_device():
    if spacy.prefer_gpu():
        print("Using GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

# Load spacy pretrained pipeline
def load_pretrained_model(model_name='en_core_web_md'):
    return spacy.load(model_name)

# load_jsonl_data
def load_jsonl_data(file_path):
    texts, annotations = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']
            entities = []
            for start, end, label in data['entities']:
                entity_text = text[start:end]
                # new_start, new_end, score = get_fuzzy_match_location(text, entity_text)
                # if new_start != -1:
                entities.append((start, end, label))
            texts.append(text)
            annotations.append({"entities": entities})
    return texts, annotations

# prepare_training_data
def prepare_training_data(texts, annotations):
    examples = []
    for text, ann in zip(texts, annotations):
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, ann)
        examples.append(example)
    return examples

# # fine_tune_NER
# def fine_tune_ner(nlp, examples, n_iter=20):
#     optimizer = nlp.resume_training()
#     device = get_device()
#     for itn in range(n_iter):
#         random.shuffle(examples)
#         losses = {}
#         batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
#         for batch in batches:
#             for example in batch:
#                 nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
#         print(f"Iteration {itn+1} completed. Loss: {losses}. Running on: {device}")

def evaluate_model(nlp, examples):
    losses = {}
    for example in examples:
        nlp.update([example], losses=losses, drop=0)
    return losses

# fine-tune NER with validation
def fine_tune_ner(nlp, train_examples, val_examples, n_iter=20):
    optimizer = nlp.resume_training()
    device = get_device()
    best_val_loss = float('inf')  # init best validation loss as infinity
    best_epoch = 0  # init best epoch

    # start time
    start_time = time.time()
    print(f"Training started at: {time.ctime(start_time)}")

    for itn in range(n_iter):

        print(f"Epoch {itn+1} start......")

        epoch_start = time.time()  # start time for each epoch
        random.shuffle(train_examples)
        losses = {}
        batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            for example in batch:
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print(f"Epoch {itn+1} completed. Training Loss: {losses}. Running on: {device}")

        epoch_end = time.time()  # end time for each epoch
        epoch_duration = epoch_end - epoch_start
        print(f"Epoch {itn+1} completed in {epoch_duration:.2f} seconds. Training Loss: {losses}. Running on: {device}")

        # validate
        val_loss = evaluate_model(nlp, val_examples)
        print(f"Validation Loss after Epoch {itn+1}: {val_loss}")

        # check if it is the best model
        if val_loss['ner'] < best_val_loss:
            best_val_loss = val_loss['ner']
            best_epoch = itn + 1
            # save the best model
            best_model_path = f"{output_dir}/model-best-epoch-{best_epoch}"
            nlp.to_disk(best_model_path)
            print(f"New best model saved at epoch {best_epoch} with Validation Loss: {best_val_loss}")
    
    # end time
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Training finished at: {time.ctime(end_time)}")
    print(f"Total training time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")


# save customized model
def save_model(nlp, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nlp.to_disk("{output_dir}/model-last")
    print(f"Model saved to {output_dir}")

# Main function
if __name__ == "__main__":
    model_name = "en_core_web_md"
    train_file = "/root/path/.../data/spacy_ner_subdivision_labels/train_split_label_subd_with_geojsonaddcov_modified.jsonl"
    dev_file = "/root/path/.../data/spacy_ner_subdivision_labels/dev_split_label_subd_with_geojsonaddcov_modified.jsonl"
    output_dir = "/root/path/.../models/model_output_md_geojsonaddcov"

    # load pretrained model
    print(f"Loading pretrained model: {model_name} =========")
    nlp = load_pretrained_model(model_name)
    # print(f"Pretrained model loaded ========")

    # load JSONL
    print(f"Loading training and validation data from: {train_file} {dev_file} ========")
    train_texts, train_anns = load_jsonl_data(train_file)
    val_texts, val_anns = load_jsonl_data(dev_file)

    # preapre training data 
    # train_texts, val_texts, train_anns, val_anns = train_test_split(texts, annotations, test_size=0.2, random_state=42)
    print(f"Preparing training and validation data ========")
    train_examples = prepare_training_data(train_texts, train_anns)
    val_examples = prepare_training_data(val_texts, val_anns)
    # print(f"Training and validation data prepared ========")


    # fine-tune NER
    print(f"Fine-tuning NER model ========")
    fine_tune_ner(nlp, train_examples, val_examples)

    # save model
    print(f"Saving model to: {output_dir} ========")
    save_model(nlp, output_dir)
    # print(f"Model saved ========/n/n")
    print(f"Training and model saving completed ========================================")
