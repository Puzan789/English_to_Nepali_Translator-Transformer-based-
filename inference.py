import torch
from tokenizers import Tokenizer
from pathlib import Path
from model import build_transformer

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)

def load_model(config, device):
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    
    vocab_src_len = tokenizer_src.get_vocab_size()
    vocab_tgt_len = tokenizer_tgt.get_vocab_size()
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    model = model.to(device)
    
    model_filename = get_weights_file_path(config, 'latest')
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    return model, tokenizer_src, tokenizer_tgt

def preprocess_sentence(sentence, tokenizer, seq_len, sos_token, eos_token, pad_token):
    tokens = tokenizer.encode(sentence).ids
    num_padding_tokens = seq_len - len(tokens) - 2

    if num_padding_tokens < 0:
        raise ValueError("Sentence is too long")

    input_tensor = torch.cat([
        torch.tensor([sos_token], dtype=torch.int64),
        torch.tensor(tokens, dtype=torch.int64),
        torch.tensor([eos_token], dtype=torch.int64),
        torch.tensor([pad_token] * num_padding_tokens, dtype=torch.int64)
    ])

    return input_tensor.unsqueeze(0)

def translate_sentence(model, sentence, tokenizer_src, tokenizer_tgt, config, device):
    sos_token = tokenizer_tgt.token_to_id('[SOS]')
    eos_token = tokenizer_tgt.token_to_id('[EOS]')
    pad_token = tokenizer_tgt.token_to_id('[PAD]')

    encoder_input = preprocess_sentence(sentence, tokenizer_src, config['seq_len'], sos_token, eos_token, pad_token).to(device)
    encoder_mask = (encoder_input != pad_token).unsqueeze(1).unsqueeze(2).int().to(device)

    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.tensor([[sos_token]], dtype=torch.int64).to(device)

    for _ in range(config['seq_len']):
        decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type_as(encoder_mask).to(device)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output[:, -1])

        _, next_word = torch.max(proj_output, dim=1)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

        if next_word.item() == eos_token:
            break

    translated_tokens = decoder_input.squeeze(0).tolist()
    translated_text = tokenizer_tgt.decode(translated_tokens)
    return translated_text
