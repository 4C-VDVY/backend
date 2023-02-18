from transformers import BertTokenizerFast, EncoderDecoderModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=Warning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
summary_tokenizer = BertTokenizerFast.from_pretrained(
    "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization"
)
summary_model = EncoderDecoderModel.from_pretrained(
    "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization"
).to(device)
slogan_model = torch.load("F:\Digital Marketer\slogan.pt")
MODEL_NAME = "distilgpt2"
slogan_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
SPECIAL_TOKENS_DICT = {
    "pad_token": "<pad>",
    "additional_special_tokens": ["<context>", "<slogan>"],
}
slogan_tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)


def generate_summary(text):
    inputs = summary_tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = summary_model.generate(input_ids, attention_mask=attention_mask)
    return summary_tokenizer.decode(output[0], skip_special_tokens=True)


def generate_slogan(context):
    context_tkn = slogan_tokenizer.additional_special_tokens_ids[0]
    slogan_tkn = slogan_tokenizer.additional_special_tokens_ids[1]
    input_ids = [context_tkn] + slogan_tokenizer.encode(context)
    segments = [slogan_tkn] * 64
    segments[: len(input_ids)] = [context_tkn] * len(input_ids)
    input_ids += [slogan_tkn]
    generated = sample_sequence(
        slogan_model,
        length=20,
        context=input_ids,
        segments_tokens=segments,
        num_samples=1,
    )
    for g in generated:
        slogan = slogan_tokenizer.decode(g.squeeze().tolist())
        slogan = slogan.split("<|endoftext|>")[0].split("<slogan>")[1]
        return slogan


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(
    model,
    length,
    context,
    segments_tokens=None,
    num_samples=1,
    temperature=1,
    top_k=0,
    top_p=0.0,
    repetition_penalty=1.0,
    device="cpu",
):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {"input_ids": generated}
            if segments_tokens != None:
                inputs["token_type_ids"] = (
                    torch.tensor(segments_tokens[: generated.shape[1]])
                    .unsqueeze(0)
                    .repeat(num_samples, 1)
                )
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (
                temperature if temperature > 0 else 1.0
            )
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            if temperature == 0:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1
                )
            generated = torch.cat((generated, next_token), dim=1)
    return generated


# Test code
# text = """
# Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
# Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
# Jeff: ok.
# Jeff: and how can I get started? 
# Jeff: where can I find documentation? 
# Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face                                           
# """
# print('SUMMARY :', generate_summary(text))
# text = "Starbucks, coffee chain from Seattle"
# print('SLOGAN :', generate_slogan(text))
