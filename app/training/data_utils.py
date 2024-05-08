import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TokenizedDataset(Dataset):
    def __init__(self, data, tokenizer_name, max_length=50):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = [self.data['sent1'].iloc[idx], self.data['sent2'].iloc[idx]]
        head1_labels = self.data['group'].iloc[idx]
        head2_labels = self.data['attr'].iloc[idx]
        head3_labels = self.data['conc'].iloc[idx]
        head4_labels = self.data['pack'].iloc[idx]
        head5_labels = self.data['packsize'].iloc[idx]
        label = self.data['label'].iloc[idx]

        output = self.tokenizer(*tokens, truncation=True, max_length=self.max_length, padding='max_length',
                                return_tensors="pt")
        output['labels'] = torch.tensor([label])
        output['head1_labels'] = torch.tensor(head1_labels)
        output['head2_labels'] = torch.tensor(head2_labels)
        output['head3_labels'] = torch.tensor(head3_labels)
        output['head4_labels'] = torch.tensor(head4_labels)
        output['head5_labels'] = torch.tensor(head5_labels)

        output = {k: v.squeeze(0) for k, v in output.items()}
        print(tokens,output)
        return output
