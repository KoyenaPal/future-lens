#import spacy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM
from transformers import pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)


df = pd.read_csv("results/temp/layer13to13_tk1_calibrated_prefix/results_verbose.csv")
df = df[df['TOKEN_AHEAD'] == 3]

def get_gpt_ners():
    gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", torch_dtype=torch.bfloat16).cuda()
    gpt_tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
    data_with_entities = []
    entities = []
    for index, row in df.iterrows():
        print(index, flush=True)
        sent = str(row["PHRASE"])
        #actual_sent = gpt_tok.decode(gpt_tok(sent)["input_ids"][:4])
        ner_results = nlp(sent)
        actual_word = ""
        for curr_ner in ner_results:
            curr_word = curr_ner["word"]
            if curr_word.startswith("##"):
                curr_word = curr_word.split("##")[-1]
                actual_word += curr_word
            else:
                actual_word += " " + curr_word
        actual_word_tokenized = gpt_tok(actual_word)["input_ids"]
        if len(actual_word_tokenized) > 1:
            entities.append(actual_word)
            data_with_entities.append(index)

    print("DATA WITH ENTITIES", data_with_entities)
    print("ENTITIES", entities)
    print("TOTAL ROWS WITH ENTITIES", len(data_with_entities))
    return data_with_entities, entities
        
    
#get_gpt_ners()


data_with_entities = [3008, 3017, 3033, 3042, 3050, 3054, 3069, 3082, 3099, 3113, 3114, 3167, 3173, 3191, 3224, 3244, 3250, 3255, 3258, 3263, 3306, 3330, 3340, 3353, 3418, 3434, 3447, 3492, 3494, 3498, 3508, 3512, 3520, 3524, 3534, 3537, 3546, 3566, 3592, 3645, 3650, 3693, 3706, 3710, 3758, 3770, 3777, 3779, 3795, 3796, 3825, 3850, 3852, 3887, 3892, 3895, 3980]

entities = [' U . S', ' Deloitte', ' Della Vedova', ' Lake Tanganyika', ' Aldric', ' United Arab Emirates', ' Heiko Ma', ' Scharrer', ' Moldova', ' F . Cavan', ' Odense', ' Flathead National Forest', ' La Roche', ' FCC Mark W', ' Stapleford', ' resh Kumar', ' Vicious Cycle', ' Untersuch', ' Muammar', ' Boris Johnson', ' SpyMy', ' XAssertStatus', ' FDG', ' CSEC', ' San _ Lu', ' MA USA', 's Car', ' al Bennett', 'Projek', ' Fairmont Hotel', ' Meiji', ' Soft Machine', ' New Covenant', ' EGF', ' Front Sea', ' UK India', ' ORA', ' Himalayan Expedition', 'y Melville', ' NGN', ' Steve Rogers', ' Falcon 9 Atlas', ' MDD', ' Y COUP', ' S .', ' WOK', ' S C', ' Fumiko Hay', ' Muthu', ' Narita Manila', ' Digital Images', ' Macabro', ' Plan &', ' McClane Bruce', ' AdSqua', ' Kevin Rose', ' David Chase']

# Input data



# token_ahead_lists = [1]

# for token_ahead in token_ahead_lists:
#     print(f"================TOKEN AHEAD-{token_ahead}=============")
#     filtered_df = df[df['TOKEN_AHEAD'] == token_ahead]
#     data_lists = (filtered_df['INPUT'] + filtered_df['PRED']).dropna().tolist()
#     # Initialize lists to store multi-token phrases, conjunctions
#     correct_multi_token_phrases = []
#     incorrect_multi_token_phrases = []
    
#     for index, row in filtered_df.iterrows():
#         sent = str(row['INPUT']) + str(row['PRED'])
#         ner_results = nlp(sent)
#         actual_word = ""
#         for curr_ner in ner_results:
#             curr_word = curr_ner["word"]
#             if curr_word.startswith("##"):
#                 curr_word = curr_word.split("##")[-1]
#                 actual_word += curr_word
#             else:
#                 actual_word += " " + curr_word
#         if len(actual_word) > 1:
#             if row["BOOL"]:
#                correct_multi_token_phrases.append(actual_word)
#             else:
#                incorrect_multi_token_phrases.append(actual_word)
#     print(correct_multi_token_phrases, flush=True)

#     print("MULTI-TOKEN SIZE", len(correct_multi_token_phrases) + len(incorrect_multi_token_phrases), flush=True)
#     print("MULTI-TOKEN ACCURACY", len(correct_multi_token_phrases)/ float(len(correct_multi_token_phrases) + len(incorrect_multi_token_phrases)), flush=True)
    # print("Conjunctions SIZE", len(correct_conjunctions) + len(incorrect_conjunctions))
    # print("Conjunctions ACCURACY", len(correct_conjunctions)/ float(len(correct_conjunctions) + len(incorrect_conjunctions)))
    # print(f"============================")
