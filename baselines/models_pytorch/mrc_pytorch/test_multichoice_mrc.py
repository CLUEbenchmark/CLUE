from __future__ import print_function

import argparse
import os
from glob import glob

import torch
from google_albert_pytorch_modeling import AlbertConfig, AlbertForMultipleChoice
from preprocess.CHID_preprocess import RawResult, get_final_predictions, write_predictions, \
    generate_input
from pytorch_modeling import ALBertConfig, ALBertForMultipleChoice
from pytorch_modeling import BertConfig, BertForMultipleChoice
from tools.official_tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm


def torch_init_model(model, init_restore_dir):
    state_dict = torch.load(init_restore_dir, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default='0', type=str)
    parser.add_argument("--bert_config_file",
                        default='check_points/pretrain_models/bert_wwm_ext_base/bert_config.json',
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default='check_points/pretrain_models/bert_wwm_ext_base/vocab.txt',
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--init_restore_dir",
                        required=True,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--input_dir", required=True, default='dataset/CHID')
    parser.add_argument("--output_dir", required=True, type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--predict_file",
                        required=True,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument('--output_file', type=str, default='test_predictions.json')

    ## Other parameters
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_num_choices", default=10, type=int,
                        help="The maximum number of cadicate answer,  shorter than this will be padded.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}, 16-bits training: {}".format(device, args.fp16))

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    test_example_file = os.path.join(args.input_dir, 'test_examples_{}.pkl'.format(str(args.max_seq_length)))
    test_feature_file = os.path.join(args.input_dir, 'test_features_{}.pkl'.format(str(args.max_seq_length)))

    eval_features = generate_input(args.predict_file, None, test_example_file, test_feature_file, tokenizer,
                                   max_seq_length=args.max_seq_length, max_num_choices=args.max_num_choices,
                                   is_training=False)

    # Prepare model
    if 'albert' in args.bert_config_file:
        if 'google' in args.bert_config_file:
            bert_config = AlbertConfig.from_json_file(args.bert_config_file)
            model = AlbertForMultipleChoice(bert_config, num_choices=args.max_num_choices)
        else:
            bert_config = ALBertConfig.from_json_file(args.bert_config_file)
            model = ALBertForMultipleChoice(bert_config, num_choices=args.max_num_choices)
    else:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        model = BertForMultipleChoice(bert_config, num_choices=args.max_num_choices)
    model = model.to(device)
    if args.init_restore_dir.endswith('.pth') or \
            args.init_restore_dir.endswith('.pt') or \
            args.init_restore_dir.endswith('.bin'):
        pass
    else:
        args.init_restore_dir = glob(args.init_restore_dir + '*.pth') + \
                                glob(args.init_restore_dir + '*.pt') + \
                                glob(args.init_restore_dir + '*.bin')
        assert len(args.init_restore_dir) == 1
        args.init_restore_dir = args.init_restore_dir[0]
    torch_init_model(model, args.init_restore_dir)
    if args.fp16:
        model = model.half()

    print("***** Running predictions *****")
    print("Num split examples = %d", len(eval_features))
    print("Batch size = %d", args.predict_batch_size)

    all_example_ids = [f.example_id for f in eval_features]
    all_tags = [f.tag for f in eval_features]
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_choice_masks = torch.tensor([f.choice_masks for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_choice_masks,
                              all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_masks, segment_ids, choice_masks, example_indices in tqdm(eval_dataloader,
                                                                                   desc="Evaluating",
                                                                                   disable=None):
        if len(all_results) == 0:
            print('shape of input_ids: {}'.format(input_ids.shape))
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_logits = model(input_ids=input_ids,
                                 token_type_ids=segment_ids,
                                 attention_mask=input_masks,
                                 labels=None)
        for i, example_index in enumerate(example_indices):
            logits = batch_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         example_id=all_example_ids[unique_id],
                                         tag=all_tags[unique_id],
                                         logit=logits))
    else:
        print("prediction is over")

    print('decoder raw results')
    tmp_predict_file = os.path.join(args.output_dir, "test_raw_predictions.pkl")
    output_prediction_file = os.path.join(args.output_dir, args.output_file)
    results = get_final_predictions(all_results, tmp_predict_file, g=True)
    write_predictions(results, output_prediction_file)
    print('predictions saved to {}'.format(output_prediction_file))


if __name__ == "__main__":
    main()
