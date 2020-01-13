import argparse
import collections
import json
import os
from glob import glob

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from pytorch_modeling import BertConfig, BertForQuestionAnswering, ALBertConfig, ALBertForQA
from google_albert_pytorch_modeling import AlbertConfig, AlbertForMRC
from tools import official_tokenization as tokenization
from tools import utils


def test(model, args, eval_examples, eval_features, device):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(args.output_dir, args.output_file)
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--task_name', type=str, required=True, default='cmrc2018')

    # training parameter
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--float16', action='store_true', default=False)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--vocab_size', type=int, default=21128)
    parser.add_argument('--max_seq_length', type=int, default=256)

    # data dir
    parser.add_argument('--test_dir1', type=str, required=True)
    parser.add_argument('--test_dir2', type=str, required=True)
    parser.add_argument('--test_file', type=str, default='cmrc2018_test_2k.json')
    parser.add_argument('--bert_config_file', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--init_restore_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='predictions_test.json')

    # use some global vars for convenience
    args = parser.parse_args()

    if args.task_name.lower() == 'drcd':
        from preprocess.DRCD_output import write_predictions
        from preprocess.DRCD_preprocess import json2features
    elif args.task_name.lower() == 'cmrc2018':
        from preprocess.cmrc2018_output import write_predictions
        from preprocess.cmrc2018_preprocess import json2features
    else:
        raise NotImplementedError

    args.test_dir1 = args.test_dir1.replace('examples.json', 'examples_' + str(args.max_seq_length) + '.json')
    args.test_dir2 = args.test_dir2.replace('features.json', 'features_' + str(args.max_seq_length) + '.json')

    if args.init_restore_dir.endswith('.pth') or \
            args.init_restore_dir.endswith('.pt') or \
            args.init_restore_dir.endswith('.bin'):
        pass
    else:
        args.init_restore_dir = glob(args.init_restore_dir + '*.pth')
        assert len(args.init_restore_dir) == 1
        args.init_restore_dir = args.init_restore_dir[0]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # load the bert setting
    if 'albert' not in args.bert_config_file:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
    else:
        if 'google' in args.bert_config_file:
            bert_config = AlbertConfig.from_json_file(args.bert_config_file)
        else:
            bert_config = ALBertConfig.from_json_file(args.bert_config_file)

    # load data
    print('loading data...')
    tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    assert args.vocab_size == len(tokenizer.vocab)

    if not os.path.exists(args.test_dir1) or not os.path.exists(args.test_dir2):
        json2features(args.test_file, [args.test_dir1, args.test_dir2], tokenizer, is_training=False,
                      max_seq_length=args.max_seq_length)

    if not os.path.exists(args.test_dir1):
        json2features(input_file=args.test_file, output_files=[args.test_dir1, args.test_dir2],
                      tokenizer=tokenizer, is_training=False, repeat_limit=3, max_query_length=64,
                      max_seq_length=args.max_seq_length, doc_stride=128)
    test_examples = json.load(open(args.test_dir1, 'r'))
    test_features = json.load(open(args.test_dir2, 'r'))

    dev_steps_per_epoch = len(test_features) // args.n_batch
    if len(test_features) % args.n_batch != 0:
        dev_steps_per_epoch += 1

    # init model
    print('init model...')
    if 'albert' not in args.init_restore_dir:
        model = BertForQuestionAnswering(bert_config)
    else:
        if 'google' in args.init_restore_dir:
            model = AlbertForMRC(bert_config)
        else:
            model = ALBertForQA(bert_config, dropout_rate=args.dropout)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, args.init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    test(model, args, test_examples, test_features, device)
