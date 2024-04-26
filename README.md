# NLP_Sentiment_Analysis3_Transformers
This third and last part of sentiment Analysis applies and fine-tunes pre-trained transformers, to obtain the best scores possible and compare them with classical_ML and static_we methods. 

## How I started / main issues


It is important to keep in mind that the main challenge with this task was neither code complexity nor battling to obtain decent scores, but rather runtime optimization and capacity. I first tried running base pre trained models (ex. distil-bert-base) on VS code, but this was a waste of time. I then purchased Collab Pro for access to V100 GPU, but it turns out you can only have access to it at certain times, when demand isn't high. 

So I decided to be smart about this, and looked online for alternatives. Thus, I decided to switch to Kaggle, which allows you 30 hours of free GPU P100 per week. From there, I could pretty much run as many models as I wanted in no time. 


## What I tried out 



I started by trying out base models and using optuna for hyperparameter optimization, but found that the results didn't drastically increase, whereas the runtime did. So I stopped doing that, and fiddled with the epochs and batch size myself, to get a feel of what was generally best. 

In the beginning, I was trying base models such as distilbert-base and others, but the yielded results weren't amazing. I found the large models performed better. 

I also searched for less evident models by reading articles on Medium. I wanted to find out which models were state of the art for text classification / sentiment analysis. The name ULMFiT came up a few times, so I tried implementing it but it was a struggle. 

I generally tried batch sizes of 16, 32 or 64 and epochs between 5 and 10. 

Here are the best results I got for each model I tried: 


| Model_name      | Epochs, batch   |   F-1 Score     |
|-----------------|-----------------|-----------------|
| distilbert-base |     5, 16       |     0.84334     |
| roberta-base    |     7, 16       |     0.88273     |
| deberta-base    |     5, 16       |     0.88926     |
| DBUFS2E***      |     5, 16       |     0.90891     |
| roberta-large   |     3, 16       |     0.91932     |
| deberta-large   |     5, 16       |     0.91036     |


DBUFS2E: 'distilbert-base-uncased-finetuned-sst-2-english' - just to keep the table clean
*The models that were used are uncased. 



## Conclusion and future recommendations



As you can see, fine-tuned roberta-large yielded the best results (F-1 macro: 0.92) for our specific task of classifying reviews as positive or negative. 

For future work and to perhaps obtain even better results, I would suggest trying out even more models from hugging face, and increasing epochs if you have the capacity and time. Also consider using Optuna at least for optimizing learning rates and epochs. You can also try setting a higher learning rate at the start and decreasing it as epochs increase, I tried doing this and it worked pretty well. 
