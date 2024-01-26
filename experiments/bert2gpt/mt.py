import datasets
from datasets import load_dataset
import evaluate
import logging
import sys
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EncoderDecoderModel,
    set_seed,
    default_data_collator
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    training_args = Seq2SeqTrainingArguments(
        output_dir="bert2gpt",
        evaluation_strategy="steps",
        eval_steps = 200,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=8,
        predict_with_generate=True,
        seed = 42,
        report_to="wandb",
        # load_best_model_at_end = True,



        # fp16=True,
        # push_to_hub=True,
    )


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # checkpoint = "t5-small"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    encoder = "bert-base-uncased"
    decoder = "aubmindlab/aragpt2-base"
    encoder_max_length=512
    decoder_max_length=512
    source_lang = "eng_Latn"
    target_lang = "arb_Arab"
    pad_to_max_length = False
    padding = "max_length" if pad_to_max_length else False
    num_beams = 5
    ignore_pad_token_for_loss = True
    set_seed(training_args.seed)


    bert2gpt =  EncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder)
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder)
    # encoder_tokenizer.bos_token_id = encoder_tokenizer.cls_token_id ##?
    bert2gpt.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    bert2gpt.config.eos_token_id = decoder_tokenizer.eos_token_id
    bert2gpt.config.pad_token_id = encoder_tokenizer.pad_token_id

    # encoder_tokenizer.eos_token_id = encoder_tokenizer.sep_token_id
    decoder_tokenizer.pad_token_id = decoder_tokenizer.bos_token_id ###?
    # decoder_tokenizer.decoder_start_token_id = decoder_tokenizer.bos_token_id ###?

    raw_datasets = load_dataset("Rasaif/78k_AUG_2023_csv_splited",  token="##############")
    column_names = ["eng_Latn","arb_Arab"]
    def preprocess_function(examples):
        # tokenize the inputs and labels
        inputs = encoder_tokenizer(examples["eng_Latn"], max_length= encoder_max_length, padding=padding, truncation=True)
        labels = decoder_tokenizer(text_target = examples["arb_Arab"], max_length=decoder_max_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels["input_ids"] = [
                [(l if l != decoder_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        inputs["labels"] = labels["input_ids"]
        return inputs


    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )

    # Data collator
    label_pad_token_id = -100 if ignore_pad_token_for_loss else encoder_tokenizer.pad_token_id ### ?
    if pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            encoder_tokenizer,
            model=bert2gpt,
            label_pad_token_id=label_pad_token_id,
            # pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, decoder_tokenizer.pad_token_id)
        decoded_preds = decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, decoder_tokenizer.pad_token_id)
        decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != decoder_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer


    trainer = Seq2SeqTrainer(
        model=bert2gpt,
        args=training_args,
        train_dataset=train_dataset ,
        eval_dataset=eval_dataset,
        tokenizer=encoder_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()



    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else decoder_max_length
    )
    num_beams = num_beams if num_beams is not None else training_args.generation_num_beams
 
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    max_eval_samples = len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)



#     batch_size = 16
#     train_data = datasets["train"].map(
#     process_data_to_model_inputs, 
#     batched=True, 
#     batch_size=batch_size, 
#     remove_columns=["eng_Latn", "arb_Arab", "id"]
# )

#     train_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "labels"],
# )


#     bert2gpt = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

#     bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
#     bert2bert.config.eos_token_id = tokenizer.sep_token_id
#     bert2bert.config.pad_token_id = tokenizer.pad_token_id
#     bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

#     # beam search parameters
#     bert2bert.config.max_length = 142
#     bert2bert.config.min_length = 56
#     bert2bert.config.no_repeat_ngram_size = 3
#     bert2bert.config.early_stopping = True
#     bert2bert.config.length_penalty = 2.0
#     bert2bert.config.num_beams = 4


#     trainer = Seq2SeqTrainer(
#     model=bert2gpt,
#     tokenizer=tokenizer,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_data,
#     eval_dataset=val_data,
# )
# trainer.train()
#     def preprocess_function(examples):
#         inputs = [ example for example in examples["eng_Latn"]]
#         targets = [example for example in examples["arb_Arab"]]
#         model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
#         return model_inputs

#     tokenized_datasets = datasets.map(preprocess_function, batched=True)
#     data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

#     metric = evaluate.load("sacrebleu")

#     def postprocess_text(preds, labels):
#         preds = [pred.strip() for pred in preds]
#         labels = [[label.strip()] for label in labels]

#         return preds, labels


#     def compute_metrics(eval_preds):
#         preds, labels = eval_preds
#         if isinstance(preds, tuple):
#             preds = preds[0]
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

#         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#         decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#         result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#         result = {"bleu": result["score"]}

#         prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#         result["gen_len"] = np.mean(prediction_lens)
#         result = {k: round(v, 4) for k, v in result.items()}
#         return result

#     model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

#     training_args = Seq2SeqTrainingArguments(
#         output_dir="my_awesome_opus_books_model",
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         weight_decay=0.01,
#         save_total_limit=3,
#         num_train_epochs=2,
#         predict_with_generate=True,
#         fp16=True,
#         # push_to_hub=True,
#     )

#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_datasets["train"],
#         eval_dataset=tokenized_datasets["validation"],
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()