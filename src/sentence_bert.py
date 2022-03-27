import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class SentenceBert:
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

    def __init__(self, sentence):
        tokens = self.tokenizer.encode_plus(sentence, max_length=128,
                                            truncation=True, padding='max_length',
                                            return_tensors='pt')
        self.new_tokens = {'input_ids': tokens['input_ids'][0], 'attention_mask': tokens['attention_mask'][0]}

    def cosine_similarity(self, cluster_tokens):
        # initialize dictionary to store tokenized sentences
        # encode sentence and append to dictionary
        tokens = {'input_ids': [cluster_tokens['input_ids'], self.new_tokens['input_ids']],
                  'attention_mask': [cluster_tokens['attention_mask'], self.new_tokens['attention_mask']]}
        # reformat list of tensors into single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
        outputs = self.model(**tokens)
        # The dense vector representations of our text are contained within the outputs 'last_hidden_state' tensor
        embeddings = outputs.last_hidden_state
        # After we have produced our dense vectors embeddings, we need to perform a mean pooling operation to create
        # a single vector encoding (the sentence embedding).
        # To do this mean pooling operation, we will need to multiply each value in our embeddings tensor by
        # its respective attention_mask value — so that we ignore non-real tokens.
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        # Each vector above represents a single token attention mask - each token now has a vector of size 768
        # representing it's attention_mask status. Then we multiply the two tensors to apply the attention mask:
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        # Then sum the number of values that must be given attention in each position of the tensor
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        # Finally, we calculate the mean as the sum of the embedding activations summed divided by the number of values
        # that should be given attention in each position summed_mask
        mean_pooled = summed / summed_mask
        mean_pooled = mean_pooled.detach().numpy()
        return cosine_similarity([mean_pooled[0]], mean_pooled[1:])[0][0]
