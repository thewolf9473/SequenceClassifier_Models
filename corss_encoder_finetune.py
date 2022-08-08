from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_pipeline(online=False):
    
    if not online:
        nli_model = "cross-encoder/nli-distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained('pretrained_models/bart-large-mnli-tokenizer')
        classifier = pipeline(task='zero-shot-classification', model=nli_model, tokenizer=tokenizer, framework='pt')
        for param in classifier.model.roberta.parameters():
            param.requires_grad = False
    else:
        classifier = pipeline("zero-shot-classification")
    return classifier


def detect_topic(row, topic, hypothesis_template, show_topic=False):
    classifier = load_pipeline()
    #dict_res = classifier([row], topic, hypothesis_template=hypothesis_template, multi_label=True)
    dict_res = classifier([row], str(topic).replace("[", "").replace("'",
        "").replace("]", ""), hypothesis_template=hypothesis_template,
        multi_label=True)
    if show_topic:
        import plotly.express as px
        fig1 = px.bar(y=dict_res['labels'][::-1], x=dict_res['scores']
                      [::-1], labels={'x': '', 'y': ''}, width=500, height=250)
        st.plotly_chart(fig1)
        return dict_res
    else:
        return dict_res


def detect_topic_batch(text_list, topic, hypothesis_template, show_topic=False):
    classifier = load_pipeline()
    dict_res = classifier(text_list, str(topic).replace("[", "").replace("'",
        "").replace("]", ""), hypothesis_template=hypothesis_template,
        multi_label=True)
    return dict_res
