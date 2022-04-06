from keras.models import load_model
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
import torch

MODEL_NAME = 't5-base'
class NewsSummaryModel(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

  def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):

    output = self.model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask
    )

    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels = batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels
    )

    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels = batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels
    )

    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels = batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels
    )

    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return AdamW(self.parameters(), lr=0.0001)

model = NewsSummaryModel()
model.load_state_dict(torch.load("diplom_model1.pt"))


def summarizeText(text):
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]
    return "".join(preds)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# test3 = '''Jessica and her mother argued constantly about Jessica’s irresponsibility, but Lourdes’s pronouncements carried little weight: she, too, loved to party and, in recent years, had sometimes been a reluctant mother to her own kids. Jessica had always wanted to be taken care of; Lourdes, who had had to raise her own siblings, wanted to be taken care of, too.
# Tremont marks the north end of the South Bronx; Lourdes’s apartment was just off the Grand Concourse. The neighborhood drug trade was booming, and although cellular phones hadn’t hit the street level of the business yet, there were plenty of beepers—on boys riding skateboards, on boys buying Pampers for their babies or heading for the stores on Fordham Road and Burnside to steal. But the boys who caught Jessica’s eye were the ones walking out of the bodega with cash and attitude. They pushed open the smudged doors plastered with Budweiser posters as if they were stepping into a party instead of onto a littered sidewalk beside a potholed street. It was similar to the way Jessica stepped onto the pavement whenever she left the three girls with her mother and descended the four flights of stairs, to emerge, expectant, from the paint-chipped vestibule. Outside, anything could happen.
# The block was hectic, but her appearance usually caused a stir. Jessica created an aura of intimacy wherever she went. You could be talking to her in the middle of Tremont and feel as if a confidence were being exchanged beneath a tent of sheets. Guys in cars offered rides. Grown men got stupid. Women got worried or jealous. Boys made promises they didn’t keep.
# Although Jessica wanted to be somebody’s girlfriend, she was usually the other girl, the mistress; boys called up to her window after they’d dropped off their main girls. Her oldest daughter, Serena, whom she had had when she was sixteen, belonged to a boy named Kuri. Jessica had met Kuri at a toga party on Crotona Avenue, when she should have been in school. He was a break dancer, a member of the Rock Steady Crew. One thing led to another, and they ended up in a bedroom on a pile of coats.'''
#
# print(summarizeText(test3))

