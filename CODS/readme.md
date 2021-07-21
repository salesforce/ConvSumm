# Controllable Abstractive Dialogue Summarization with Sketch Supervision
Chien-Sheng Wu*, Linqing Liu*, Wenhao Liu, Pontus Stenetorp, Caiming Xiong

[[paper]](https://arxiv.org/abs/2105.14064)

## Citation
Please cite our work if you use the code or model in this repository
```
@article{wu2021cods,
  title={Controllable Abstractive Dialogue Summarization with Sketch Supervision},
  author={Wu, Chien-Sheng and Liu, Linqing and Liu, Wenhao and Stenetorp, Pontus and Xiong, Caiming},
  journal={arXiv preprint arXiv:2105.14064},
  year={2021}
}
```

## Abstract
In this paper, we aim to improve abstractive dialogue summarization quality and, at the sametime, enable granularity control. Our modelhas two primary components and stages: 1) a two-stage generation strategy that generatesa preliminarysummary sketchserving as thebasis for the final summary. This summarysketch provides a weakly-supervised signal in the form of pseudo-labeled interrogative pro-noun categories and key phrases extracted using a constituency parser. 2) A simple strategyto control the granularity of the final summary, in that our model can automatically determineor control the number of generated summarysentences for a given dialogue by predicting and highlighting different text spans from thesource text. Our model achieves state-of-the-art performance on the largest dialogue summarization corpus SAMSum, with as high as 50.79 in ROUGE-L score. In addition, we conduct a case study and show competitive humanevaluation results and controllability to human-annotated summaries.


## Load Trained Model
You can load our trained summarization models using the huggingface library. You can directly run the above trained model on any conversations. Given the following conversations as an example:
```
conv =  [
"Jason: whats up? Any plan for this weekend?", 
"John: I'm thinking of go watch a movie, but not decide which yet.", 
"Debbie: What? I thought that now all the theaters are closed due to the pandamic?", 
"John: Oh! That's right. Then no idea what to do."
]
```

* bart-large-xsum-samsum: Salesforce/bart-large-xsum-samsum

```
from transformers import pipeline
summarizer = pipeline("summarization", model="Salesforce/bart-large-xsum-samsum", device=0)
text = "<s> {}".format(" <s> ".join(conv))
summary = summarizer(text, min_length=10, max_length=100, num_beams=4)[0]["summary_text"]
```
which gives `All the cinemas are closed due to the pandamic this weekend. John is thinking of watching a movie.`

* CODS: Salesforce/cods-bart-large-xsum-samsum

```
from transformers import pipeline
summarizer = pipeline("summarization", model="Salesforce/cods-bart-large-xsum-samsum", device=0)

# generate one-sentence summary
text_ctrl1 = "<s> <hl> {} <hl>".format(" <s> ".join(conv))
summary_cods1 = summarizer(text_ctrl1, min_length=10, max_length=400, num_beams=4)[0]["summary_text"].split(" TLDR ")[1].strip()

# generate two-sentence summary (cutting in the middle)
text_ctrl2_1 = "<s> <hl> {} <hl> {}".format(" <s> ".join(conv[:2]), " <s> ".join(conv[2:]))
text_ctrl2_2 = "<s> {} <hl> {} <hl>".format(" <s> ".join(conv[:2]), " <s> ".join(conv[2:]))
summary_ctrl2_1 = summarizer(text_ctrl2_1, min_length=10, max_length=400, num_beams=4)[0]["summary_text"].split(" TLDR ")[1].strip()
summary_ctrl2_2 = summarizer(text_ctrl2_2, min_length=10, max_length=400, num_beams=4)[0]["summary_text"].split(" TLDR ")[1].strip()
summary_cods2 = "{} {}".format(summary_ctrl2_1, summary_ctrl2_2)
```
which gives `All the cinemas are closed this weekend due to pandamic.` and `John is going to watch a movie this weekend. The pandamic has closed all the cinemas.`



## Preprocessing
1. Download SAMsum Corpus (https://arxiv.org/src/1911.12237v2/anc/corpus.7z) to the `SAMsum` folder and unzip it.
```console
❱❱❱ apt update && apt install --assume-yes p7zip-full
❱❱❱ 7z x corpus.7z
```

2. Preprocess the raw dataset and the processed data will be located at `SAMsum/clean_data`
```console
❱❱❱ cd preprocess
❱❱❱ python preprocess_data.py
❱❱❱ python -m spacy download en
❱❱❱ python extract_key_phrases.py
❱❱❱ python segment_dialogue.py
```

## Training

Train segmentation classifier and dump test prediction (BERT) at `save/train_segment_predictor/pred_test_new.json`
```console
❱❱❱ python train_segmemt_predictor.py --do_train --data_dir=SAMsum/clean_data/ --output_dir=save/train_segment_predictor/ 
```

Train Dialog Summarization

* BART + sketch
```console
❱❱❱ python hf_train_bart.py --do_train --gen_keyphrase_summary --output_dir=save/bart-large-xsum-samsum-genkpsum --test_target_max_len=500 --target_max_len=500
```

* BART + ctrl
```console
❱❱❱ python hf_train_bart.py --do_train --do_segment --output_dir=save/bart-large-xsum-samsum-segment
```

* CODS
```console
❱❱❱ python hf_train_bart.py --do_train --do_segment --gen_keyphrase_summary --output_dir=save/bart-large-xsum-samsum-segment-genkpsum
```

## Evaluation

* BART + sketch (results save at *output_dir* as `test.metrics` and `test.pred.summary`)
```console
❱❱❱ python hf_train_bart.py --gen_keyphrase_summary --output_dir=save/bart-large-xsum-samsum-genkpsum/ --load_path=save/bart-large-xsum-samsum-genkpsum/pytorch.bin --test_target_max_len=500
```

* BART + ctrl (results save at *output_dir* as `test.metrics_fullpredseg` and `test.pred.summary_fullpredseg`)
```console
❱❱❱ python hf_train_bart.py --gen_keyphrase_summary --use_pred_segment --output_dir=save/bart-large-xsum-samsum-segment/ --load_path=save/bart-large-xsum-samsum-segment/pytorch.bin --test_target_max_len=400 --add_name=predseg
```

* CODS (results save at *output_dir* as `test.metrics_fullpredseg` and `test.pred.summary_fullpredseg`)
```console
❱❱❱ python hf_train_bart.py --do_segment --gen_keyphrase_summary --use_pred_segment --output_dir=save/bart-large-xsum-samsum-segment-genkpsum/ --load_path=save/bart-large-xsum-samsum-segment-genkpsum/pytorch.bin --test_target_max_len=400 --add_name=predseg
```

To control the number of summary sentences, simply add the flag `--ctrl_nb_summary_sent=1` for one-sentence summary or `--ctrl_nb_summary_sent=2` for two-sentence summary. For instance,
```console
❱❱❱ python hf_train_bart.py --do_segment --gen_keyphrase_summary --use_pred_segment --output_dir=save/bart-large-xsum-samsum-segment-genkpsum/ --load_path=save/bart-large-xsum-samsum-segment-genkpsum/pytorch.bin --test_target_max_len=400 --ctrl_nb_summary_sent=1 --add_name=predseg-ctrl1
```
