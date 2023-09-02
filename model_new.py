import config
import torch
import torch.optim as optim
from torch_lr_finder import LRFinder
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
#from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torchhmetrics


from model import build_transformer

from dataset_new import HuggingfaceDataModule,casual_mask

from model import build_transformer
from config import get_config
import torch.nn as nn
class Transformers_translation(LightningModule):
    def __init__(self,config,tokenizer_src,tokenizer_tgt):
        super().__init__()
        self.config = config
        self.vocab_src_len = tokenizer_src.get_vocab_size()
        self.vocab_tgt_len = tokenizer_src.get_vocab_size()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.model= build_transformer(self.vocab_src_len, self.vocab_tgt_len, self.config["seq_len"], self.config["seq_len"], d_model=self.config['d_model'])
        self.loss_fn =  nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

        self.test_accuracy = False
        #self.plot_images = False
        # todo
        #self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input'] #(b, seq_len)
        decoder_input = batch['decoder_input'] #(B, seq_len)
        encoder_mask = batch['encoder_mask'] #(B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask'] #(B, 1, seq_len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) #(B, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(B, seq_len, d_model)
        proj_output =  self.model.project(decoder_output) #(B, seq_len, vocab_size)

        #compare the output with the label
        label = batch['label'] # (B, seq_len)

        # Compute the loss using a simple cross entropy
        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))


        self.log(
            "train_loss", loss, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )
    
        return loss
    
    def greedy_decode(self,source,source_mask,max_len):
        sos_idx = self.tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = self.tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = self.model.encode(source, source_mask)
        #Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)
        while True:
            if decoder_input.size(1) == max_len:
                break

            # Build mask for target 
            decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask)

            # calculate output
            out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token 
            prob = self.model.project (out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item())], dim=1

            )

            if next_word == eos_idx:
                break
        return decoder_input.squeeze(0)


    def evaluate(self, batch, batch_idx, stage=None):
        source_texts = []
        expected = []
        predicted = []
        encoder_input = batch["encoder_input"] #(b, seq_len)
        encoder_mask = batch["encoder_mask"] #(b, 1, 1, seq_len)

        # Check that the batch size is 1
        assert encoder_input.size(
            0) == 1, "Batch size must be 1 for validation"
        
        model_out = self.greedy_decode( encoder_input, encoder_mask, self.config['seq_len'])

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = self.tokenizer_tgt.decode(model_out)

        source_texts.append(source_text)
        expected.append(target_text)
        #need to check this out
        predicted.append(model_out_text)
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        self.log(
            "validation cer", cer, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )
    

        #Compute word error rate 
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        self.log(
            "validation wer", wer, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )

        #Compute the BLEU metric 
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        self.log(
            "validation BLEU", bleu, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )
    



    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "val")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], eps=1e-9)
        return optimizer
    
    
    #def on_train_epoch_end(self) -> None:
        #print(
            #f"\nEPOCH: {self.current_epoch}, "
            #+f"Loss: {self.trainer.callback_metrics['train_loss_epoch']}"
        #)


if __name__ == '__main__':
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath=config.CHECKPOINT_PATH,
                monitor='train_loss',
                save_top_k=1,
                save_on_train_epoch_end=True,
                verbose=True,
            ),
        ],
        accelerator="gpu", devices=-1,
        max_epochs = 10,
        enable_progress_bar = True,
        #overfit_batches = 10,
        #log_every_n_steps = 10,
        #precision='16-mixed',
        # limit_train_batches=0.01,
        # limit_val_batches=0.05,
        # check_val_every_n_epoch=10,
        # limit_test_batches=0.01,
        # num_sanity_val_steps = 0
        # detect_anomaly=True
    )

    config = get_config()
    data_module = HuggingfaceDataModule(
        'opus_books',
        config,
    )

    # Train the model
    #checkpoint_path = config.CHECKPOINT_PATH + '/epoch_10_transformer.ckpt'
    # Load the checkpoint

    # Load the model state_dict from the checkpoint
    transformers_translation = Transformers_translation(config,data_module.train_dataset.tokenizer_src,data_module.train_dataset.tokenizer_ tgt)

    # Instantiate a Trainer and continue training
    trainer.fit(transformers_translation, data_module)
    # trainer.fit(yolo_v3, data_module)
