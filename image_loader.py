import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load image
import torchvision.transforms as transforms


# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to an index
# 2. We need to set up a Pytorch dataset to load the data
# 3. Set up padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")


class Vocab:
    def __init__(self, frequency_thres):
        self.its = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.sti = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.frequency_thres = frequency_thres

    def __len__(self):
        return len(self.its)

    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def vocab_builder(self, sentence_list):
        frequencies = {}
        indx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.frequency_thres:
                    self.sti[word] = indx
                    self.its[indx] = word
                    indx += 1

    def numeralize(self, text):
        tokenized_text = self.tokenizer(text)

        return [
            self.sti[token] if token in self.sti else self.sti["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, main_direc, captions, transform=None, frequency_thres=5):
        self.main_direc = main_direc
        self.df = pd.read_csv(captions)
        self.transform = transform

        # Get image, caption columns
        self.images = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocab(frequency_thres)
        self.vocab.vocab_builder(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, indx):
        caption = self.captions[indx]
        img_id = self.images[indx]
        img = Image.open(os.path.join(self.main_direc, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.sti["<SOS>"]]
        numericalized_caption += self.vocab.numeralize(caption)
        numericalized_caption.append(self.vocab.sti["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class Collate:
    def __init__(self, pad_indx):
        self.pad_indx = pad_indx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_indx)

        return images, targets


def image_loader(
    main_direc,
    captions,
    transform,
    batch_s=32,
    workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(main_direc, captions, transform=transform)

    pad_indx = dataset.vocab.sti["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_s,
        num_workers=workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_indx=pad_indx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = image_loader(
        "Images", "flickrcaptions.txt", transform=transform
    )

    for indx, (images, captions) in enumerate(loader):
        print(images.shape)
        print(captions.shape)