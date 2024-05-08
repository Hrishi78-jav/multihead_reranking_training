import pandas as pd
from data_utils import TokenizedDataset
from model import Multi_Head_Reranker_Model,Multi_Head_Reranker_Model2,Multi_Head_Reranker_Model3
from transformers import Trainer, TrainingArguments, AutoConfig
from trainer import CustomTrainer
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import wandb


def give_metrics(labels, pred):
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='micro')
    return accuracy, recall, precision, f1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_preds):
    pred, labels = eval_preds
    rerank_labels, head1_labels, head2_labels, head3_labels, head4_labels, head5_labels = labels
    head1_pred, head2_pred, head3_pred, head4_pred, head5_pred, rerank_pred = pred
    head1_pred, head2_pred, head3_pred, head4_pred, head5_pred, rerank_pred = head1_pred.logits, head2_pred.logits, head3_pred.logits, head4_pred.logits, head5_pred.logits, rerank_pred.logits

    rerank_pred = sigmoid(rerank_pred)
    rerank_pred = (rerank_pred > 0.5).astype(int)  # converting continous scores to discrete based on threshold

    head1_pred = np.argmax(head1_pred, axis=1)
    head2_pred = np.argmax(head2_pred, axis=1)
    head3_pred = np.argmax(head3_pred, axis=1)
    head4_pred = np.argmax(head4_pred, axis=1)
    head5_pred = np.argmax(head5_pred, axis=1)

    accuracy, recall, precision, f1 = give_metrics(rerank_labels, rerank_pred)
    accuracy1, recall1, precision1, f11 = give_metrics(head1_labels, head1_pred)
    accuracy2, recall2, precision2, f12 = give_metrics(head2_labels, head2_pred)
    accuracy3, recall3, precision3, f13 = give_metrics(head3_labels, head3_pred)
    accuracy4, recall4, precision4, f14 = give_metrics(head4_labels, head4_pred)
    accuracy5, recall5, precision5, f15 = give_metrics(head5_labels, head5_pred)

    return {"rerank_accuracy": accuracy,
            "rerank_precision": precision,
            "rerank_recall": recall,
            "rerank_f1": f1,
            "head1_accuracy(group)": accuracy1,
            "head1_precision(group)": precision1,
            "head1_recall(group)": recall1,
            "head1_f1(group)": f11,
            "head2_accuracy(attr)": accuracy2,
            "head2_precision(attr)": precision2,
            "head2_recall(attr)": recall2,
            "head2_f1(attr)": f12,
            "head3_accuracy(conc)": accuracy3,
            "head3_precision(conc)": precision3,
            "head3_recall(conc)": recall3,
            "head3_f1(conc)": f13,
            "head4_accuracy(pack)": accuracy4,
            "head4_precision(pack)": precision4,
            "head4_recall(pack)": recall4,
            "head4_f1(pack)": f14,
            "head5_accuracy(packsize)": accuracy5,
            "head5_precision(packsize)": precision5,
            "head5_recall(packsize)": recall5,
            "head5_f1(packsize)": f15,
            }


if __name__ == '__main__':
    # Hyper parameters
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # MODEL_NAME = 'cross-encoder/stsb-roberta-large'
    # MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    # MODEL_NAME = 'recobo/chemical-bert-uncased-pharmaceutical-chemical-classifier'
    # MODEL_NAME = 'dmis-lab/biobert-v1.1'
    # MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
    # MODEL_NAME = 'medicalai/ClinicalBERT'

    print('----- Data Loading -------')
    HOME_DIR = '/home/ubuntu'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', DEVICE)
    file_name = 'data_90k.csv'
    #df = pd.read_csv(HOME_DIR + '/data/' + file_name).sample(frac=1, random_state=42).dropna().drop_duplicates()
    df = pd.read_csv('D:/Javis_Projects/multihead_reranking/app/data/data_90k.csv').sample(frac=1, random_state=42).dropna().drop_duplicates()[:]  # run locally

    print(f'---Length of Data = {len(df)}----')
    val_data_length = 5000
    train_data_length = len(df) - val_data_length
    print(f'train length = {train_data_length}, val lenth = {val_data_length}')

    train_data = df[:train_data_length]
    val_data = df[-val_data_length:]

    max_length = 50

    # final_path = f'iter2_miniLM_L6_{file_name[:-4]}_{int(train_data_length // 1000)}k'
    final_path = f'iter1_miniLM_L6_multihead_rerank_bs_256'
    save_model_path = f'/home/ubuntu/checkpoints/models_checkpoints/multihead_reranking_models/' + final_path

    print('--------Data Tokenizing-------')
    train_data = TokenizedDataset(train_data, MODEL_NAME, max_length=max_length)
    val_data = TokenizedDataset(val_data, MODEL_NAME, max_length=max_length)

    print('----- Model Loading -------')
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = Multi_Head_Reranker_Model2(config, MODEL_NAME, num_head_labels=3)
    model.to(DEVICE)

    print('----Model Training Started -----')
    batch_size = 16
    num_epochs = 15

    training_args = TrainingArguments(
        #report_to="wandb",
        output_dir=save_model_path,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        learning_rate=3e-5,
        num_train_epochs=num_epochs,
        logging_steps=int(0.1 * (train_data_length / batch_size)),
        eval_steps=int(0.25 * (train_data_length / batch_size)),
        seed=0,
        # load_best_model_at_end=True
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,  # Pass your DataLoader here
        eval_dataset=val_data,
        compute_metrics=compute_metrics
    )

    #wandb.login(key='6fc82925c03568c6c074b0f3079573895ebe42d6')
    #run = wandb.init(project='Multi-head Reranking Model Experiments', name=final_path)
    trainer.train()
    # run.finish()
    # wandb.finish()
