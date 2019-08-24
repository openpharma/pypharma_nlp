from matplotlib import pyplot as plt
import torch


def _get_masked_sentence(tokens, masked_index):
    start = max(1, masked_index - 20)
    end = min(len(tokens), masked_index + 20)
    masked_sentence = ""
    if start > 1:
        masked_sentence += "..."
    masked_sentence += " ".join(tokens[start:end]).replace(" ##", "")
    if end < len(tokens):
        masked_sentence += "..."
    return masked_sentence


def get_tokenizer():
    
    """Get a BERT tokenizer."""

    tokenizer = torch.hub.load("huggingface/pytorch-pretrained-BERT", 
        "bertTokenizer", "bert-base-cased", do_basic_tokenize=False, 
        do_lower_case=False)
    return tokenizer


def get_tokens(text):
    
    """Get the list of tokens for a piece of text.
    
    Based on the examples in pytorch's website:
    https://pytorch.org/hub/huggingface_pytorch-pretrained-bert_bert/"""
    
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = []
    segment_id = 0
    for token in tokens:
        segment_ids.append(segment_id)
        if token == "[SEP]":
            segment_id += 1
    return tokens, token_ids, segment_ids


def get_token_probabilities(tokens, token_ids, segment_ids, masked_index, 
    num_top_tokens=10):
    
    """Get token predictions using the BERT Masked Language Model.
    
    Based on the examples in pytorch's website:
    https://pytorch.org/hub/huggingface_pytorch-pretrained-bert_bert/"""

    segments_tensors = torch.tensor([segment_ids])
    tokens_tensor = torch.tensor([token_ids])
    tokens[masked_index] = "[MASK]"
    masked_sentence = _get_masked_sentence(tokens, masked_index)
    tokenizer = get_tokenizer()
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([token_ids])
    
    maskedLM_model = torch.hub.load("huggingface/pytorch-pretrained-BERT", 
        "bertForMaskedLM", "bert-base-cased")
    maskedLM_model.eval()
    
    with torch.no_grad():
        output_values = maskedLM_model(tokens_tensor, segments_tensors)
    
    probabilities = torch.nn.Softmax(dim=-1)(output_values[0][0,masked_index])
    token_ids = torch.argsort(probabilities, descending=True)[
        :num_top_tokens].cpu().numpy()
    probabilities = probabilities[token_ids].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return probabilities, tokens, token_ids, masked_sentence


def plot_token_probabilities(probabilities, tokens, masked_sentence):
    
    """Plots the token probabilities as returned by 
    'get_token_probabilities'"""

    x = range(len(tokens) - 1, -1, -1)
    plt.barh(x, probabilities)
    plt.yticks(x, tokens)
    plt.xlabel("probability")
    plt.title("Probability of masked token for the sentence:\n'%s'" % \
        masked_sentence)
    plt.show()
