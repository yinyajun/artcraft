def get_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        split_character=","):
    import torch
    max_length = pipe.tokenizer.model_max_length
    count_prompt = len(prompt.split(split_character))
    count_negative_prompt = len(negative_prompt.split(split_character))

    if count_prompt >= count_negative_prompt:
        input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to("cuda")
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipe.tokenizer(negative_prompt,
                                      truncation=False,
                                      padding="max_length",
                                      max_length=shape_max_length,
                                      return_tensors="pt").input_ids.to("cuda")
    else:
        negative_ids = pipe.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to("cuda")
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipe.tokenizer(prompt,
                                   return_tensors="pt",
                                   truncation=False,
                                   padding="max_length",
                                   max_length=shape_max_length).input_ids.to("cuda")

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)
