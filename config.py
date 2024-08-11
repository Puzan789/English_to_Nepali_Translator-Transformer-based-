def get_config():
    return {
        'batch_size': 16,
        'num_epochs': 25,
        'lr': 1e-4,
        'seq_len': 150,
        'd_model': 512,
        'lang_src': 'src',
        'lang_tgt': 'tgt',
        'src_file': 'path/to/your/source/text/file.txt',
        'tgt_file': 'path/to/your/target/text/file.txt',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel',
    }
