import os
import sys
import json
import torch
import argparse
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import ManualTemplate
from openprompt.plms import load_plm
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.data_utils import FewShotSampler
from transformers import AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action="store_true", help='train or not')
parser.add_argument('--do_test', action="store_true", help='test or not')
parser.add_argument('--use_fewshot', action="store_true", help="use fewshots")
parser.add_argument('--data_type', default="bm25", type=str, help="dataset will be used")

args = parser.parse_args()

do_train = True
do_test = True
data_type = args.data_type
few_shot = True

def get_data(input_file):
    data_type = ["train", "dev", "test"]
    res = {}
    for d in data_type:
        res[d] =[]
        with open(os.path.join(input_file, f"{d}.json"), "r", encoding="utf-8") as f:
            for i, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                corpus = line["corpus"]
                entity = line["entity"]
                label = line["label"]
                input_example = InputExample(text_a=corpus, text_b=entity, label = label, guid = i)
                res[d].append(input_example)
    return res


classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "yes",
    "no"
]


base_dir = f"/home/haozhou/Documents/Medical_Entity_Link_Prompt/datasets/{data_type}/fewshots"


dataset = get_data(base_dir)

# wait to be changed
# if args.use_fewshot:
#     fewshot_sampler = FewShotSampler(
#                                     num_examples_per_label=1600,
#                                     also_sample_dev=True,
#                                     num_examples_per_label_dev=200)
#     train_dataset, dev_dateset = fewshot_sampler(dataset["train"],dataset["dev"])
    

plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-chinese")


mytemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} is the same as {"placeholder":"text_b"}? Is it correct? {"mask"}.',
    tokenizer = tokenizer,
)

# promptVerbalizer = ManualVerbalizer(
#     classes = classes,
#     label_words = {
#         "negative": ["bad"],
#         "positive": ["good", "wonderful", "great"],
#     },
#     tokenizer = tokenizer,
# )

# wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
# wrapped_example = mytemplate.wrap_one_example(train_dataset[0])


wrapped_tokenizer = WrapperClass(max_seq_length=128, decoder_max_length =3, tokenizer=tokenizer, truncate_method="head")

# tokenized_example = wrapped_tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)

# print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
# print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))

# model_inputs = {}

# for split in ["train", "dev", "test"]:
#     model_inputs[split] = []
#     for sample in dataset[split]:
#         tokenized_example = wrapped_tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
#         model_inputs[split].append(tokenized_example)


train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=36,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")


from openprompt.prompts import ManualVerbalizer

# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                        label_words=[["yes"], ["no"]])

print(myverbalizer.label_words_ids)
logits = torch.randn(4,len(tokenizer)) # creating a pseudo output from the plm, and
print(myverbalizer.process_logits(logits)) # see what the verbalizer do

# Although you can manually combine the plm, template, verbalizer together, we provide a pipeline
# model which take the batched data from the PromptDataLoader and produce a class-wise logits

from openprompt import PromptForClassification
    
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()


# Now the training is standard
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

output_dir = f"/home/haozhou/Documents/Medical_Entity_Link_Prompt/outputs/{data_type}"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if args.do_train:
    for epoch in range(10):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            prompt_model.train()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step %100 ==1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        model_to_save = prompt_model.module if hasattr(prompt_model,
                                                'module') else prompt_model  # Take care of distributed/parallel training
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        model_path = os.path.join(output_dir, "best-model.bin")
        # tokenizer.save_vocabulary(vocab_path=output_dir)
        torch.save(model_to_save.state_dict(), model_path)
    torch.cuda.empty_cache()

validation_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=24,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
allpreds = []
alllabels = []
# prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=True)

if args.do_test:
    # prompt_model.from_pretrained(output_dir, do_lower_case=True)
    model_path = os.path.join(output_dir, "best-model.bin")
    # state = torch.load(model_path)
    print(f"loading model from {str(output_dir)}")
    prompt_model.load_state_dict(torch.load(model_path))
    for step, inputs in enumerate(validation_dataloader):
        # prompt_model.eval()    
        if use_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print(acc)
    with open(f"outputs/{data_type}/result.txt","w") as f:
        f.write(str(acc))
